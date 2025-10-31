#!/usr/bin/env python3
"""
3D Gaussian Splatting集成模块
用于将实时稀疏点云数据传递给3DGS进行稠密重建
"""

import os
import sys
import torch
import numpy as np
import threading
import time
from typing import List, Optional
import subprocess

# 添加3DGS路径
sys.path.append("thirdparty/gaussian-splatting")

try:
    # 设置CUDA扩展所需的环境变量
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
    torch_lib_path = '/home/surgicalai/anaconda3/envs/ELGS/lib/python3.10/site-packages/torch/lib'
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] += ':' + torch_lib_path
    else:
        os.environ['LD_LIBRARY_PATH'] = torch_lib_path
        
    from utils.graphics_utils import BasicPointCloud
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss, ssim
    from utils.general_utils import safe_state
    import math
    
    GAUSSIAN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: 3D Gaussian Splatting not available: {e}")
    GAUSSIAN_AVAILABLE = False

class VirtualCamera:
    """简化的虚拟相机类"""
    def __init__(self, width=400, height=300, fov=60, position=None, target=None, camera_id=0):
        self.image_width = width
        self.image_height = height
        self.image_name = f"virtual_camera_{camera_id:03d}"
        
        # 计算视场角
        self.FoVx = math.radians(fov)
        self.FoVy = 2 * math.atan(math.tan(self.FoVx / 2) * height / width)
        
        # 设置相机位置和朝向
        if position is None:
            position = np.array([0, 0, 3])
        if target is None:
            target = np.array([0, 0, 0])
            
        self.camera_center = torch.tensor(position, dtype=torch.float32).cuda()
        
        # 构建视图矩阵
        up = np.array([0, 1, 0])
        forward = target - position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # 视图矩阵 (world to camera)
        R = np.array([right, up, -forward])
        t = -R @ position
        
        view_matrix = np.eye(4)
        view_matrix[:3, :3] = R
        view_matrix[:3, 3] = t
        
        self.world_view_transform = torch.tensor(view_matrix, dtype=torch.float32).cuda().T
        
        # 投影矩阵
        znear = 0.01
        zfar = 100.0
        
        proj_matrix = np.zeros((4, 4))
        proj_matrix[0, 0] = 1 / math.tan(self.FoVx / 2)
        proj_matrix[1, 1] = 1 / math.tan(self.FoVy / 2)
        proj_matrix[2, 2] = zfar / (zfar - znear)
        proj_matrix[2, 3] = -(zfar * znear) / (zfar - znear)
        proj_matrix[3, 2] = 1
        
        self.projection_matrix = torch.tensor(proj_matrix, dtype=torch.float32).cuda().T
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix

class GaussianTrainer:
    """3DGS训练器，用于从稀疏点云进行稠密重建"""
    
    def __init__(self, output_dir=None, pixel_scale=0.001):
        # 如果没有提供输出目录，使用临时目录或不创建目录
        if output_dir is None:
            self.output_dir = None  # 实时模式不需要输出目录
        else:
            self.output_dir = output_dir
            # 只在提供了输出目录时才创建
            os.makedirs(output_dir, exist_ok=True)
            
        self.pixel_scale = pixel_scale
        self.gaussians = None
        self.cameras = None
        self.training_thread = None
        self.is_training = False
        self.latest_model_path = None
        
    def prepare_point_cloud(self, points_3d, colors):
        """将3D点云和颜色转换为BasicPointCloud格式"""
        if len(points_3d) == 0:
            return None
            
        # 确保points_3d是numpy数组
        if isinstance(points_3d, list):
            points_3d = np.array(points_3d)
        if isinstance(colors, list):
            colors = np.array(colors)
            
        # 应用像素缩放
        points = points_3d * self.pixel_scale
        
        # 颜色归一化到[0,1]
        if colors.max() > 1.0:
            colors = colors / 255.0
            
        # 中心化点云
        center = np.mean(points, axis=0)
        points = points - center
        
        # 创建零法向量
        normals = np.zeros_like(points)
        
        return BasicPointCloud(points=points, colors=colors, normals=normals)
    
    def create_virtual_cameras(self, point_cloud, num_cameras=4, radius_scale=2.0):
        """围绕点云创建虚拟相机"""
        points = np.array(point_cloud.points)
        center = np.mean(points, axis=0)
        max_dist = np.max(np.linalg.norm(points - center, axis=1))
        radius = max(max_dist * radius_scale, 0.5)
        
        cameras = []
        for i in range(num_cameras):
            angle = 2 * math.pi * i / num_cameras
            elevation = math.pi / 6  # 30度仰角
            
            x = center[0] + radius * math.cos(elevation) * math.cos(angle)
            y = center[1] + radius * math.sin(elevation)
            z = center[2] + radius * math.cos(elevation) * math.sin(angle)
            
            position = np.array([x, y, z])
            camera = VirtualCamera(width=400, height=300, fov=45, 
                                 position=position, target=center, camera_id=i)
            cameras.append(camera)
        
        return cameras
    
    def train_async(self, points_3d, colors, iterations=5000):
        """异步训练3DGS模型"""
        if not GAUSSIAN_AVAILABLE:
            print("3D Gaussian Splatting not available")
            return False
            
        if self.is_training:
            print("Training already in progress")
            return False
            
        # 准备数据
        point_cloud = self.prepare_point_cloud(points_3d, colors)
        if point_cloud is None:
            print("No valid point cloud data")
            return False
            
        print(f"Starting 3DGS training with {len(points_3d)} points...")
        
        # 启动训练线程
        self.training_thread = threading.Thread(
            target=self._train_worker,
            args=(point_cloud, iterations),
            daemon=True
        )
        self.training_thread.start()
        return True
    
    def _train_worker(self, point_cloud, iterations):
        """训练工作线程"""
        self.is_training = True
        
        try:
            # 创建虚拟相机
            self.cameras = self.create_virtual_cameras(point_cloud, num_cameras=4)
            
            # 初始化高斯模型
            self.gaussians = GaussianModel(sh_degree=3)
            
            # 计算场景范围
            points = np.array(point_cloud.points)
            scene_extent = np.max(np.linalg.norm(points - np.mean(points, axis=0), axis=1))
            
            # 从点云创建高斯
            self.gaussians.create_from_pcd(point_cloud, self.cameras, scene_extent)
            
            # 设置训练参数
            class TrainingArgs:
                def __init__(self, max_iterations):
                    self.iterations = max_iterations
                    self.position_lr_init = 0.00016
                    self.position_lr_final = 0.0000016
                    self.position_lr_delay_mult = 0.01
                    self.position_lr_max_steps = max_iterations
                    self.feature_lr = 0.0025
                    self.opacity_lr = 0.05
                    self.scaling_lr = 0.005
                    self.rotation_lr = 0.001
                    self.densify_from_iter = 500
                    self.densify_until_iter = min(15000, max_iterations//2)
                    self.densify_grad_threshold = 0.0002
                    self.densification_interval = 100
                    self.opacity_reset_interval = 3000
                    self.lambda_dssim = 0.2
                    self.percent_dense = 0.01
                    # 添加缺少的exposure相关参数
                    self.exposure_lr_init = 0.001
                    self.exposure_lr_final = 0.0001
                    self.exposure_lr_delay_steps = 5000
                    self.exposure_lr_delay_mult = 0.001
                
            class PipelineArgs:
                convert_SHs_python = False
                compute_cov3D_python = False
                debug = False
                antialiasing = False
            
            training_args = TrainingArgs(iterations)
            self.gaussians.training_setup(training_args)
            
            # 训练循环
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            
            print(f"Training 3DGS for {iterations} iterations...")
            
            for iteration in range(1, iterations + 1):
                self.gaussians.update_learning_rate(iteration)
                
                # 随机选择相机
                camera = self.cameras[iteration % len(self.cameras)]
                
                # 渲染
                render_pkg = render(camera, self.gaussians, 
                                  pipe=PipelineArgs(), 
                                  bg_color=bg_color)
                image = render_pkg["render"]
                
                # 检查可见性
                visibility_filter = render_pkg["visibility_filter"]
                if len(visibility_filter) == 0:
                    continue
                
                # 创建简单目标（自监督训练）
                gt_image = image.detach()
                if iteration > 50:
                    noise = torch.randn_like(gt_image) * 0.01
                    gt_image = torch.clamp(gt_image + noise, 0, 1)
                
                # 计算损失
                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - training_args.lambda_dssim) * Ll1 + training_args.lambda_dssim * (1.0 - ssim(image, gt_image))
                
                loss.backward()
                
                with torch.no_grad():
                    # 密度控制
                    if (iteration >= training_args.densify_from_iter and 
                        iteration <= training_args.densify_until_iter and 
                        len(visibility_filter) > 0):
                        
                        self.gaussians.max_radii2D[visibility_filter] = torch.max(
                            self.gaussians.max_radii2D[visibility_filter], 
                            render_pkg["radii"][visibility_filter]
                        )
                        self.gaussians.add_densification_stats(render_pkg["viewspace_points"], visibility_filter)
                        
                        if iteration % training_args.densification_interval == 0:
                            size_threshold = 20 if iteration > training_args.opacity_reset_interval else None
                            self.gaussians.densify_and_prune(training_args.densify_grad_threshold, 
                                                            0.005, scene_extent, size_threshold, 
                                                            render_pkg["radii"])
                        
                        if iteration % training_args.opacity_reset_interval == 0:
                            self.gaussians.reset_opacity()
                    
                    # 优化器步进
                    if iteration < iterations:
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none=True)
                
                # 进度报告
                if iteration % 500 == 0:
                    print(f"3DGS Training: {iteration}/{iterations}, Loss: {loss.item():.4f}")
            
            # 保存模型
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_filename = f"gaussian_model_{timestamp}.ply"
            self.latest_model_path = os.path.join(self.output_dir, model_filename)
            self.gaussians.save_ply(self.latest_model_path)
            
            print(f"3DGS training completed! Model saved: {model_filename}")
            
        except Exception as e:
            print(f"3DGS training error: {e}")
        finally:
            self.is_training = False
    
    def get_latest_model(self):
        """获取最新训练的模型路径"""
        return self.latest_model_path
    
    def is_training_active(self):
        """检查是否正在训练"""
        return self.is_training
    
    def render_view(self, camera_position=None, camera_target=None):
        """从指定视角渲染高斯模型"""
        if self.gaussians is None or not GAUSSIAN_AVAILABLE:
            return None
            
        try:
            # 创建相机
            if camera_position is None:
                camera_position = np.array([0, 0, 3])
            if camera_target is None:
                camera_target = np.array([0, 0, 0])
                
            camera = VirtualCamera(width=800, height=600, fov=45,
                                 position=camera_position, target=camera_target)
            
            # 渲染
            class PipelineArgs:
                convert_SHs_python = False
                compute_cov3D_python = False
                debug = False
                antialiasing = False
            
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            
            with torch.no_grad():
                render_pkg = render(camera, self.gaussians, 
                                  pipe=PipelineArgs(), 
                                  bg_color=bg_color)
                image = render_pkg["render"]
                
                # 转换为numpy数组
                image_np = image.detach().cpu().numpy()
                image_np = np.transpose(image_np, (1, 2, 0))  # CHW -> HWC
                image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
                
                return image_np
                
        except Exception as e:
            print(f"Rendering error: {e}")
            return None

class GaussianIntegration:
    """3DGS集成主类"""
    
    def __init__(self, output_dir="output_3dgs", pixel_scale=0.001, auto_train_threshold=500):
        self.trainer = GaussianTrainer(output_dir, pixel_scale)
        self.auto_train_threshold = auto_train_threshold
        self.accumulated_points = []
        self.accumulated_colors = []
        self.last_auto_train_count = 0  # 记录上次自动训练时的点数
        self.training_triggered = False  # 防止重复触发训练
        
    def add_points(self, points_3d, colors):
        """添加新的3D点云数据"""
        if len(points_3d) > 0:
            self.accumulated_points.extend(points_3d)
            if len(colors) > 0:
                self.accumulated_colors.extend(colors)
    
    def should_auto_train(self):
        """判断是否应该自动开始训练"""
        current_count = len(self.accumulated_points)
        # 只有当累积点数达到阈值且比上次训练时增加了足够多的点时才训练
        points_since_last_train = current_count - self.last_auto_train_count
        
        return (current_count >= self.auto_train_threshold and 
                points_since_last_train >= self.auto_train_threshold and
                not self.trainer.is_training_active() and
                not self.training_triggered)
    
    def start_training(self, iterations=5000, auto_triggered=False):
        """开始3DGS训练"""
        if len(self.accumulated_points) == 0:
            print("No accumulated points for training")
            return False
            
        if self.trainer.is_training_active():
            print("Training already in progress")
            return False
            
        # 记录训练状态
        if auto_triggered:
            self.training_triggered = True
            self.last_auto_train_count = len(self.accumulated_points)
            
        points = np.array(self.accumulated_points)
        colors = np.array(self.accumulated_colors) if len(self.accumulated_colors) > 0 else np.ones((len(points), 3)) * 128
        
        # 启动训练，完成后重置标志
        success = self.trainer.train_async(points, colors, iterations)
        
        # 训练完成回调
        if success and auto_triggered:
            # 使用线程监控训练完成
            def monitor_training():
                import time
                while self.trainer.is_training_active():
                    time.sleep(1)
                self.training_triggered = False  # 训练完成，重置标志
                print("Training completed, ready for next auto-training")
            
            threading.Thread(target=monitor_training, daemon=True).start()
        
        return success
    
    def get_status(self):
        """获取当前状态"""
        return {
            'accumulated_points': len(self.accumulated_points),
            'is_training': self.trainer.is_training_active(),
            'latest_model': self.trainer.get_latest_model(),
            'points_since_last_train': len(self.accumulated_points) - self.last_auto_train_count,
            'next_auto_train_at': self.last_auto_train_count + self.auto_train_threshold,
            'training_triggered': self.training_triggered
        }
    
    def clear_accumulated_data(self):
        """清空累积数据"""
        self.accumulated_points = []
        self.accumulated_colors = []
        self.last_auto_train_count = 0
        self.training_triggered = False
    
    def render_latest_model(self, camera_position=None, camera_target=None):
        """渲染最新的高斯模型"""
        return self.trainer.render_view(camera_position, camera_target)