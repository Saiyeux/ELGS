#!/usr/bin/env python3
"""
高斯集成线程模块
用于异步处理3D Gaussian Splatting稠密重建
"""

import time
import sys
import os
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np

# 添加3DGS路径
sys.path.append("thirdparty/gaussian-splatting")

try:
    from .gaussian_integration import GaussianIntegration, GaussianTrainer
    GAUSSIAN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Gaussian integration not available: {e}")
    GAUSSIAN_AVAILABLE = False

class GaussianThread(QThread):
    """
    3D Gaussian Splatting实时重建线程
    实时处理点云数据进行快速稠密重建
    """
    
    # 信号定义
    reconstruction_started = pyqtSignal()  # 重建开始
    reconstruction_progress = pyqtSignal(int, int, float)  # 进度更新 (current, total, loss)
    reconstruction_completed = pyqtSignal(np.ndarray)  # 重建完成 (dense_points)
    reconstruction_failed = pyqtSignal(str)  # 重建失败 (error_message)
    status_updated = pyqtSignal(dict)  # 状态更新
    dense_points_ready = pyqtSignal(np.ndarray, np.ndarray)  # 稠密点云准备好 (points, colors)
    
    def __init__(self, pixel_scale=0.001):
        super().__init__()
        
        self.pixel_scale = pixel_scale
        
        # 初始化高斯训练器
        if GAUSSIAN_AVAILABLE:
            self.gaussian_trainer = GaussianTrainer(None, pixel_scale)  # 不使用输出目录
        else:
            self.gaussian_trainer = None
        
        # 线程控制
        self.enabled = False
        self.reconstruction_iterations = 200  # 默认快速重建迭代次数
        self.is_reconstructing = False
        
        # 处理队列
        self.point_queue = []
        self.processing = False
        
        # 最新重建结果
        self.latest_dense_points = None
        self.latest_dense_colors = None
        
    def is_available(self):
        """检查高斯重建是否可用"""
        return GAUSSIAN_AVAILABLE and self.gaussian_trainer is not None
    
    def set_enabled(self, enabled):
        """启用/禁用高斯重建"""
        self.enabled = enabled
        if enabled and not self.is_available():
            print("Warning: Gaussian reconstruction is not available")
            return False
        return True
    
    def set_reconstruction_parameters(self, iterations=200, pixel_scale=None):
        """设置重建参数"""
        self.reconstruction_iterations = max(1, iterations)  # 取消上限限制，只保留最小值1
        
        if pixel_scale is not None:
            self.pixel_scale = pixel_scale
            if self.gaussian_trainer:
                self.gaussian_trainer.pixel_scale = pixel_scale
    
    def add_points(self, points_3d, colors):
        """添加新的3D点云数据进行实时重建"""
        if not self.enabled or not self.is_available():
            return
        
        if len(points_3d) == 0:
            return
        
        try:
            # 将点云数据添加到处理队列
            point_data = {
                'points': points_3d.copy(),
                'colors': colors.copy() if colors is not None else None,
                'timestamp': time.time()
            }
            
            # 只保留最新的点云数据，不进行累积
            self.point_queue = [point_data]  # 覆盖之前的数据
            
            # 启动实时重建
            if not self.processing and not self.isRunning():
                self.start()
                
        except Exception as e:
            print(f"Error adding points for Gaussian reconstruction: {e}")
    
    def start_reconstruction(self, iterations=None):
        """手动开始重建"""
        if not self.enabled or not self.is_available():
            return False
        
        if len(self.point_queue) == 0:
            return False
        
        if iterations is not None:
            self.reconstruction_iterations = max(1, iterations)  # 取消上限限制
        
        # 启动重建处理
        if not self.processing and not self.isRunning():
            self.start()
            return True
        return False
    
    def stop_reconstruction(self):
        """停止重建处理"""
        self.processing = False
        self.is_reconstructing = False
    
    def clear_queue(self):
        """清空处理队列"""
        self.point_queue.clear()
        self.processing = False
    
    def get_latest_dense_points(self):
        """获取最新的稠密点云"""
        return self.latest_dense_points, self.latest_dense_colors
    
    def emit_status_update(self):
        """发送状态更新信号"""
        status = {
            'enabled': self.enabled,
            'available': self.is_available(),
            'processing': self.processing,
            'is_reconstructing': self.is_reconstructing,
            'queue_size': len(self.point_queue),
            'reconstruction_iterations': self.reconstruction_iterations,
            'has_dense_points': self.latest_dense_points is not None
        }
        self.status_updated.emit(status)
    
    def run(self):
        """线程主循环 - 实时重建处理"""
        if not self.is_available():
            return
        
        self.processing = True
        
        try:
            while len(self.point_queue) > 0 and self.enabled:
                # 获取最新的点云数据
                point_data = self.point_queue.pop(0)
                
                # 执行快速重建
                self.reconstruct_dense_points(
                    point_data['points'],
                    point_data['colors']
                )
                
                # 短暂休眠避免CPU占用过高
                self.msleep(100)
                
        except Exception as e:
            self.reconstruction_failed.emit(str(e))
        finally:
            self.processing = False
            self.is_reconstructing = False
            self.emit_status_update()
    
    def reconstruct_dense_points(self, points_3d, colors):
        """快速重建稠密点云"""
        if not self.is_available():
            return
        
        try:
            self.is_reconstructing = True
            self.reconstruction_started.emit()
            self.emit_status_update()
            
            # 准备点云数据
            point_cloud = self.gaussian_trainer.prepare_point_cloud(points_3d, colors)
            if point_cloud is None:
                return
            
            # 创建虚拟相机
            cameras = self.gaussian_trainer.create_virtual_cameras(point_cloud, num_cameras=3)
            
            # 初始化高斯模型
            from scene.gaussian_model import GaussianModel
            gaussians = GaussianModel(sh_degree=0)  # 使用更简单的球谐函数
            
            # 计算场景范围
            points = np.array(point_cloud.points)
            scene_extent = np.max(np.linalg.norm(points - np.mean(points, axis=0), axis=1))
            
            # 从点云创建高斯
            gaussians.create_from_pcd(point_cloud, cameras, scene_extent)
            
            # 快速训练参数（针对实时性优化）
            class FastTrainingArgs:
                def __init__(self, max_iterations):
                    self.iterations = max_iterations
                    self.position_lr_init = 0.001  # 更高的学习率
                    self.position_lr_final = 0.0001
                    self.position_lr_delay_mult = 0.01
                    self.position_lr_max_steps = max_iterations
                    self.feature_lr = 0.01
                    self.opacity_lr = 0.1
                    self.scaling_lr = 0.01
                    self.rotation_lr = 0.005
                    self.densify_from_iter = max_iterations // 4  # 更早开始密度控制
                    self.densify_until_iter = max_iterations // 2
                    self.densify_grad_threshold = 0.001  # 更宽松的阈值
                    self.densification_interval = max(1, max_iterations // 20)  # 更频繁的密度控制
                    self.opacity_reset_interval = max_iterations // 2
                    self.lambda_dssim = 0.1  # 降低DSSIM权重
                    self.percent_dense = 0.1  # 更高的密度
                    self.exposure_lr_init = 0.01
                    self.exposure_lr_final = 0.001
                    self.exposure_lr_delay_steps = max_iterations // 2
                    self.exposure_lr_delay_mult = 0.01
            
            training_args = FastTrainingArgs(self.reconstruction_iterations)
            gaussians.training_setup(training_args)
            
            # 快速训练循环
            import torch
            from gaussian_renderer import render
            from utils.loss_utils import l1_loss, ssim
            
            class PipelineArgs:
                convert_SHs_python = False
                compute_cov3D_python = False
                debug = False
                antialiasing = False
            
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            
            for iteration in range(1, self.reconstruction_iterations + 1):
                if not self.enabled:  # 允许中途停止
                    break
                
                gaussians.update_learning_rate(iteration)
                
                # 随机选择相机
                camera = cameras[iteration % len(cameras)]
                
                # 渲染
                render_pkg = render(camera, gaussians, 
                                  pipe=PipelineArgs(), 
                                  bg_color=bg_color)
                image = render_pkg["render"]
                
                # 检查可见性
                visibility_filter = render_pkg["visibility_filter"]
                if len(visibility_filter) == 0:
                    continue
                
                # 简单的自监督目标
                gt_image = image.detach()
                if iteration > 10:
                    noise = torch.randn_like(gt_image) * 0.02
                    gt_image = torch.clamp(gt_image + noise, 0, 1)
                
                # 计算损失
                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - training_args.lambda_dssim) * Ll1 + training_args.lambda_dssim * (1.0 - ssim(image, gt_image))
                
                loss.backward()
                
                with torch.no_grad():
                    # 密度控制（简化版）
                    if (iteration >= training_args.densify_from_iter and 
                        iteration <= training_args.densify_until_iter and 
                        len(visibility_filter) > 0 and
                        iteration % training_args.densification_interval == 0):
                        
                        gaussians.max_radii2D[visibility_filter] = torch.max(
                            gaussians.max_radii2D[visibility_filter], 
                            render_pkg["radii"][visibility_filter]
                        )
                        gaussians.add_densification_stats(render_pkg["viewspace_points"], visibility_filter)
                        
                        size_threshold = 10 if iteration > training_args.opacity_reset_interval else None
                        gaussians.densify_and_prune(training_args.densify_grad_threshold, 
                                                  0.01, scene_extent, size_threshold, 
                                                  render_pkg["radii"])
                    
                    if iteration % training_args.opacity_reset_interval == 0:
                        gaussians.reset_opacity()
                    
                    # 优化器步进
                    if iteration < self.reconstruction_iterations:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)
                
                # 发送进度更新
                if iteration % max(1, self.reconstruction_iterations // 10) == 0:
                    self.reconstruction_progress.emit(iteration, self.reconstruction_iterations, loss.item())
            
            # 提取稠密点云
            dense_points, dense_colors = self.extract_dense_points(gaussians)
            
            if dense_points is not None and len(dense_points) > 0:
                self.latest_dense_points = dense_points
                self.latest_dense_colors = dense_colors
                
                # 发送稠密点云数据
                self.dense_points_ready.emit(dense_points, dense_colors)
                self.reconstruction_completed.emit(dense_points)
            
        except Exception as e:
            self.reconstruction_failed.emit(str(e))
        finally:
            self.is_reconstructing = False
    
    def extract_dense_points(self, gaussians):
        """从高斯模型中提取稠密点云"""
        try:
            import torch
            
            # 获取高斯中心点位置
            xyz = gaussians.get_xyz.detach().cpu().numpy()
            
            # 获取颜色 (SH coefficients -> RGB)
            features_dc = gaussians.get_features[:, :, 0:1].transpose(1, 2).detach().cpu().numpy()
            rgb = features_dc[:, 0, :] + 0.5  # 简单的SH->RGB转换
            rgb = np.clip(rgb, 0, 1) * 255
            
            # 获取不透明度
            opacity = torch.sigmoid(gaussians.get_opacity).detach().cpu().numpy()
            
            # 过滤低不透明度的点
            opacity_threshold = 0.1
            valid_mask = opacity.flatten() > opacity_threshold
            
            if np.sum(valid_mask) == 0:
                return None, None
            
            dense_points = xyz[valid_mask]
            dense_colors = rgb[valid_mask].astype(np.uint8)
            
            # 密度增强：在每个高斯点周围生成更多点
            enhanced_points = []
            enhanced_colors = []
            
            scales = torch.exp(gaussians.get_scaling).detach().cpu().numpy()[valid_mask]
            
            for i, (point, color, scale) in enumerate(zip(dense_points, dense_colors, scales)):
                # 原始点
                enhanced_points.append(point)
                enhanced_colors.append(color)
                
                # 在高斯周围生成额外的点（简化的密度增强）
                num_extra_points = min(8, max(2, int(np.mean(scale) * 100)))
                
                for _ in range(num_extra_points):
                    # 在高斯范围内随机采样
                    offset = np.random.normal(0, scale * 0.3, 3)
                    new_point = point + offset
                    enhanced_points.append(new_point)
                    enhanced_colors.append(color)
            
            return np.array(enhanced_points), np.array(enhanced_colors)
            
        except Exception as e:
            print(f"Error extracting dense points: {e}")
            return None, None
    
    def get_status_info(self):
        """获取当前状态信息（用于UI显示）"""
        if not self.is_available():
            return {
                'available': False,
                'enabled': False,
                'processing': False,
                'is_reconstructing': False,
                'reconstruction_iterations': 200,
                'has_dense_points': False
            }
        
        return {
            'available': True,
            'enabled': self.enabled,
            'processing': self.processing,
            'is_reconstructing': self.is_reconstructing,
            'queue_size': len(self.point_queue),
            'reconstruction_iterations': self.reconstruction_iterations,
            'has_dense_points': self.latest_dense_points is not None,
            'dense_point_count': len(self.latest_dense_points) if self.latest_dense_points is not None else 0
        }