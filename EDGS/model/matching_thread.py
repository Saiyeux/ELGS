#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import time
from PyQt5.QtCore import QThread, pyqtSignal
import torch
from pathlib import Path

# Import existing ELGS components
sys.path.append('thirdparty/EfficientLoFTR')
from src.loftr import LoFTR
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config
from src.loftr.loftr import reparameter

class MatchingThread(QThread):
    matches_ready = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    points_3d_ready = pyqtSignal(np.ndarray, np.ndarray)  # points_3d, colors_rgb
    log_message = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.frame0 = None
        self.frame1 = None
        self.matcher = None
        self.running = False
        self.conf_thresh = 0.2
        self.resize_factor = 0.8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Camera parameters - follow demo approach
        self.img_width, self.img_height = 1920, 1080
        
        # Use simpler camera model like demo (focal length = image width/height for stereo)
        self.fx = self.img_width  # Similar to demo's 1000 for 1280 width
        self.fy = self.img_width  # Keep square pixels
        self.cx = self.img_width / 2.0  # Principal point at center
        self.cy = self.img_height / 2.0
        
        self.camera_matrix = np.array([[self.fx, 0, self.cx],
                                     [0, self.fy, self.cy],
                                     [0, 0, 1]], dtype=np.float32)  # Use float32 like demo
        
        # Distortion coefficients (not used for essential matrix computation)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # 全局坐标系跟踪
        self.world_coordinate_initialized = False
        self.global_R = np.eye(3, dtype=np.float32)  # 全局旋转矩阵（左相机相对世界坐标系）
        self.global_t = np.zeros((3, 1), dtype=np.float32)  # 全局平移向量
        self.fixed_baseline_R = None  # 双目相机间的固定相对旋转
        self.fixed_baseline_t = None  # 双目相机间的固定相对平移
        self.baseline_samples = []  # 用于估计稳定基线的样本
        self.max_baseline_samples = 10  # 用于建立稳定基线的帧数
        
    def initialize_matcher(self):
        try:
            config = get_cfg_defaults()
            config.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
            config.LOFTR.COARSE.NPE = [832, 832, 832, 832]  # Set NPE parameter for position encoding
            config = lower_config(config)
            
            self.matcher = LoFTR(config=config['loftr'])
            
            # Load weights
            weights_path = Path('thirdparty/EfficientLoFTR/weights/eloftr_outdoor.ckpt')
            if weights_path.exists():
                state_dict = torch.load(str(weights_path), map_location='cpu', weights_only=False)['state_dict']
                self.matcher.load_state_dict(state_dict, strict=False)
                self.matcher = reparameter(self.matcher)
                self.matcher.to(self.device).eval()
                self.log_message.emit("EfficientLoFTR模型加载成功")
                self.log_message.emit(f"相机内参: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
                return True
            else:
                self.log_message.emit(f"权重文件未找到: {weights_path}")
                return False
        except Exception as e:
            self.log_message.emit(f"模型初始化失败: {str(e)}")
            return False
            
    def update_frames(self, frame0, frame1):
        self.frame0 = frame0.copy()
        self.frame1 = frame1.copy()
        
    def set_parameters(self, conf_thresh, resize_factor):
        self.conf_thresh = conf_thresh
        self.resize_factor = resize_factor
    
    def reset_world_coordinate_system(self):
        """重置世界坐标系，用于重新初始化全局跟踪"""
        self.world_coordinate_initialized = False
        self.global_R = np.eye(3, dtype=np.float32)
        self.global_t = np.zeros((3, 1), dtype=np.float32)
        self.fixed_baseline_R = None
        self.fixed_baseline_t = None
        self.baseline_samples = []
        self.log_message.emit("世界坐标系已重置")
        
    def run(self):
        if not self.matcher:
            if not self.initialize_matcher():
                return
                
        while self.running:
            if self.frame0 is not None and self.frame1 is not None:
                try:
                    # Resize frames to ensure consistent dimensions
                    h0, w0 = self.frame0.shape[:2]
                    h1, w1 = self.frame1.shape[:2]
                    
                    # Calculate target size ensuring both frames have same dimensions
                    target_h = int(min(h0, h1) * self.resize_factor)
                    target_w = int(min(w0, w1) * self.resize_factor)
                    
                    # Ensure dimensions are divisible by 8
                    target_h = (target_h // 8) * 8
                    target_w = (target_w // 8) * 8
                    
                    # Minimum size check
                    if target_h < 64 or target_w < 64:
                        target_h, target_w = 64, 64
                    
                    frame0_resized = cv2.resize(self.frame0, (target_w, target_h))
                    frame1_resized = cv2.resize(self.frame1, (target_w, target_h))
                    
                    # Convert to grayscale and normalize
                    gray0 = cv2.cvtColor(frame0_resized, cv2.COLOR_BGR2GRAY)
                    gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
                    
                    # Check if images have valid dimensions
                    if gray0.size == 0 or gray1.size == 0:
                        self.log_message.emit("错误：图像尺寸无效")
                        time.sleep(0.1)
                        continue
                    
                    # Prepare tensors - now guaranteed to have same size
                    tensor0 = torch.from_numpy(gray0)[None, None].float().to(self.device) / 255.0
                    tensor1 = torch.from_numpy(gray1)[None, None].float().to(self.device) / 255.0
                    
                    batch = {'image0': tensor0, 'image1': tensor1}
                    
                    # Run matching
                    with torch.no_grad():
                        try:
                            self.matcher(batch)
                        except RuntimeError as e:
                            if "Sizes of tensors must match" in str(e):
                                self.log_message.emit(f"张量尺寸不匹配错误: {str(e)}")
                                self.log_message.emit(f"尝试重新调整图像尺寸...")
                                # Try with square images
                                min_dim = min(tensor0.shape[2], tensor0.shape[3], tensor1.shape[2], tensor1.shape[3])
                                # Ensure divisible by 8
                                min_dim = (min_dim // 8) * 8
                                if min_dim > 0:
                                    tensor0 = torch.nn.functional.interpolate(tensor0, size=(min_dim, min_dim), mode='bilinear', align_corners=False)
                                    tensor1 = torch.nn.functional.interpolate(tensor1, size=(min_dim, min_dim), mode='bilinear', align_corners=False)
                                    batch = {'image0': tensor0, 'image1': tensor1}
                                    self.matcher(batch)
                                else:
                                    raise e
                            else:
                                raise e
                    
                    # Extract matches
                    mkpts0 = batch['mkpts0_f'].cpu().numpy()
                    mkpts1 = batch['mkpts1_f'].cpu().numpy()
                    mconf = batch['mconf'].cpu().numpy()
                    
                    # Scale back to original resolution
                    scale_x0 = w0 / target_w
                    scale_y0 = h0 / target_h
                    scale_x1 = w1 / target_w
                    scale_y1 = h1 / target_h
                    
                    mkpts0[:, 0] *= scale_x0
                    mkpts0[:, 1] *= scale_y0
                    mkpts1[:, 0] *= scale_x1
                    mkpts1[:, 1] *= scale_y1
                    
                    # Filter by confidence
                    valid = mconf > self.conf_thresh
                    mkpts0_valid = mkpts0[valid]
                    mkpts1_valid = mkpts1[valid]
                    mconf_valid = mconf[valid]
                    
                    self.matches_ready.emit(mkpts0_valid, mkpts1_valid, mconf_valid)
                    
                    # 3D reconstruction
                    if len(mkpts0_valid) >= 8:
                        self.reconstruct_3d(mkpts0_valid, mkpts1_valid)
                    
                except Exception as e:
                    self.log_message.emit(f"匹配处理错误: {str(e)}")
                    
            time.sleep(0.1)
            
    def reconstruct_3d(self, pts0, pts1):
        try:
            # Convert to numpy arrays with correct dtype
            points1_filtered = np.array(pts0, dtype=np.float32)
            points2_filtered = np.array(pts1, dtype=np.float32)
            
            if len(points1_filtered) < 5:
                return
            
            # Essential matrix estimation
            E, mask = cv2.findEssentialMat(points1_filtered, points2_filtered, self.camera_matrix, 
                                         method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            if E is None or mask is None:
                self.log_message.emit("错误：本质矩阵计算失败")
                return
            
            ransac_inliers = mask.sum()
            if ransac_inliers < 5:
                return
            
            # Recover relative pose between cameras
            _, R_rel, t_rel, pose_mask = cv2.recoverPose(E, points1_filtered, points2_filtered, 
                                                       self.camera_matrix, mask=mask)
            
            if pose_mask.sum() < 5:
                return
                
            # 建立或使用固定的世界坐标系
            if not self.world_coordinate_initialized:
                self._initialize_world_coordinate_system(R_rel, t_rel)
                return  # 第一帧用于初始化，不输出点云
            
            # 使用固定的双目基线进行三角化
            inlier_mask = mask.ravel().astype(bool)
            inlier_points1 = points1_filtered[inlier_mask]
            inlier_points2 = points2_filtered[inlier_mask]
            
            # 构建投影矩阵 - 使用固定的世界坐标系
            projMatrix1 = self._build_projection_matrix(self.global_R, self.global_t)
            projMatrix2 = self._build_projection_matrix(
                self.global_R @ self.fixed_baseline_R, 
                self.global_t + self.global_R @ self.fixed_baseline_t
            )
            
            # 三角化点云
            points_3d_homogeneous = cv2.triangulatePoints(projMatrix1, projMatrix2, 
                                                         inlier_points1.T, inlier_points2.T)
            points_3d = points_3d_homogeneous[:3] / points_3d_homogeneous[3]
            points_3d = points_3d.T
            
            # 严格的深度和异常值过滤，确保超出合理范围的点完全舍弃
            depth_threshold = 5.0  # 进一步降低深度阈值，只保留近距离的可靠点
            
            # 多重严格过滤条件
            valid_depth_mask = (
                (points_3d[:, 2] > 0.2) &  # 提高最小深度阈值
                (points_3d[:, 2] < depth_threshold) &  # 严格的最大深度阈值
                (np.abs(points_3d[:, 0]) < 2.0) &  # 收紧X方向范围
                (np.abs(points_3d[:, 1]) < 2.0) &  # 收紧Y方向范围
                np.isfinite(points_3d[:, 0]) &  # 检查数值有效性
                np.isfinite(points_3d[:, 1]) &
                np.isfinite(points_3d[:, 2]) &
                (np.abs(points_3d[:, 0]) > 0.01) &  # 排除过于接近原点的点
                (np.abs(points_3d[:, 1]) > 0.01)
            )
            
            points_3d_filtered = points_3d[valid_depth_mask]
            
            if len(points_3d_filtered) > 0:
                # 提取颜色信息
                valid_inlier_points = inlier_points1[valid_depth_mask]
                colors_rgb = self.extract_point_colors(valid_inlier_points)
                self.points_3d_ready.emit(points_3d_filtered, colors_rgb)
                
        except Exception as e:
            self.log_message.emit(f"3D重建错误: {str(e)}")
    
    def _initialize_world_coordinate_system(self, R_rel, t_rel):
        """初始化世界坐标系，建立固定的双目基线"""
        # 收集基线样本用于平均
        baseline_sample = {
            'R': R_rel.copy(),
            't': t_rel.copy()
        }
        self.baseline_samples.append(baseline_sample)
        
        if len(self.baseline_samples) >= self.max_baseline_samples:
            # 计算平均基线（简化的方法）
            # 在实际应用中，可能需要更复杂的平均方法，如旋转平均
            avg_t = np.mean([sample['t'] for sample in self.baseline_samples], axis=0)
            
            # 对于旋转矩阵，使用最后一个样本（更复杂的平均需要特殊处理）
            avg_R = self.baseline_samples[-1]['R']  # 简化处理
            
            # 设置固定基线
            self.fixed_baseline_R = avg_R.astype(np.float32)
            self.fixed_baseline_t = avg_t.astype(np.float32)
            
            # 世界坐标系原点设为第一帧左相机位置
            self.global_R = np.eye(3, dtype=np.float32)
            self.global_t = np.zeros((3, 1), dtype=np.float32)
            
            self.world_coordinate_initialized = True
            self.log_message.emit(f"世界坐标系已建立，基线长度: {np.linalg.norm(avg_t):.3f}")
        else:
            remaining = self.max_baseline_samples - len(self.baseline_samples)
            self.log_message.emit(f"正在校准双目基线... 还需 {remaining} 帧")
    
    def _build_projection_matrix(self, R, t):
        """构建投影矩阵"""
        Rt = np.hstack((R, t.reshape(-1, 1)))
        return self.camera_matrix @ Rt
    
    def extract_point_colors(self, points_2d):
        """从相机图像中提取2D点对应的RGB颜色"""
        try:
            if self.frame0 is None:
                return None
                
            colors_rgb = []
            frame_bgr = self.frame0  # OpenCV uses BGR format
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            for pt in points_2d:
                x, y = int(np.clip(pt[0], 0, frame_rgb.shape[1]-1)), int(np.clip(pt[1], 0, frame_rgb.shape[0]-1))
                # Extract RGB color at this pixel
                rgb_color = frame_rgb[y, x]  # Note: image coordinates are [y, x]
                colors_rgb.append(rgb_color)
                
            return np.array(colors_rgb, dtype=np.uint8)
        except Exception as e:
            self.log_message.emit(f"颜色提取错误: {str(e)}")
            return None
            
    def start_matching(self):
        self.running = True
        self.start()
        
    def stop_matching(self):
        self.running = False
        self.wait()