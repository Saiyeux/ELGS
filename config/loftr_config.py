#!/usr/bin/env python3
"""
EfficientLoFTR 配置管理
存储和管理EfficientLoFTR匹配相关的所有参数
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

class LoFTRConfig:
    """EfficientLoFTR配置管理类"""
    
    def __init__(self, config_file="config/loftr_params.json"):
        self.config_file = Path(config_file)
        
        # 默认配置参数
        self.defaults = {
            # 基本参数
            "enabled": True,
            "weights_path": "thirdparty/EfficientLoFTR/weights/eloftr_outdoor.ckpt",
            "model_type": "full",  # 'full' or 'opt'
            "precision": "fp32",   # 'fp32', 'fp16', 'mp' (mixed precision)
            
            # 相机参数
            "camera": {
                "camera0_id": 0,
                "camera1_id": 4,
                "img_width": 1920,
                "img_height": 1080,
                "focal_length_scale": 1.0,  # 焦距相对于图像宽度的比例
                "use_auto_focal_length": True,  # 自动计算焦距
                "manual_fx": 1920.0,
                "manual_fy": 1920.0,
                "manual_cx": 960.0,
                "manual_cy": 540.0
            },
            
            # 匹配参数
            "matching": {
                "conf_thresh": 0.2,
                "resize_factor": 0.8,
                "max_matches": 2000,
                "min_matches": 8,  # 最少匹配点数进行3D重建
                "skip_frames": 0   # 跳帧数量(0表示不跳帧)
            },
            
            # LoFTR模型配置
            "loftr_model": {
                "match_type": "dual_softmax",  # "dual_softmax" or "sinkhorn"
                "npe_config": [832, 832, 832, 832],  # NPE位置编码参数
                "coarse_resolution": [8, 2],  # 粗匹配分辨率
                "fine_resolution": 2,         # 精匹配分辨率
                "temperature": 0.1,
                "border_rm": 2,
                "match_threshold": 0.2,
                "skh_iters": 3,      # Sinkhorn迭代次数
                "skh_init_bin_score": 1.0
            },
            
            # 3D重建参数
            "reconstruction": {
                # Essential Matrix参数
                "ransac_method": "RANSAC",  # "RANSAC" or "LMEDS" 
                "ransac_confidence": 0.999,
                "ransac_threshold": 1.0,
                "max_iterations": 2000,
                
                # 姿态恢复参数
                "recover_pose_threshold": 50.0,
                "min_triangulation_angle": 1.0,  # 最小三角化角度(度)
                
                # 三角化参数
                "depth_threshold": 50.0,   # 深度过滤阈值
                "min_depth": 0.1,          # 最小深度
                "max_depth": 100.0,        # 最大深度
                "reprojection_threshold": 2.0,  # 重投影误差阈值
                
                # 点云过滤
                "enable_statistical_filter": True,
                "statistical_k": 20,
                "statistical_std_ratio": 2.0,
                "enable_radius_filter": False,
                "radius_filter_radius": 0.1,
                "radius_filter_min_neighbors": 5
            },
            
            # 显示参数
            "visualization": {
                "show_matches": True,
                "match_color": [0, 255, 0],     # BGR格式
                "match_thickness": 2,
                "confidence_color_map": True,   # 根据置信度着色
                "min_conf_color": [0, 0, 255],  # 低置信度颜色(BGR)
                "max_conf_color": [0, 255, 0],  # 高置信度颜色(BGR)
            },
            
            # 性能参数
            "performance": {
                "use_cuda": True,
                "batch_size": 1,
                "num_workers": 0,
                "device_id": 0,
                "memory_efficient": True,
                "prefetch_factor": 2
            },
            
            # 滤波集成参数
            "filtering": {
                "enable_stable_filter": True,
                "filter_before_3d": False,  # 在3D重建前过滤还是在显示前过滤
                "min_filter_ratio": 0.3,   # 最小保留比例
                "max_filter_ratio": 0.9    # 最大保留比例
            }
        }
        
        # 加载配置
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # 合并默认配置和加载的配置
                config = self.defaults.copy()
                self._deep_update(config, loaded_config)
                return config
            except Exception as e:
                print(f"Warning: Failed to load LoFTR config: {e}")
                return self.defaults.copy()
        else:
            return self.defaults.copy()
    
    def save_config(self) -> bool:
        """保存配置到文件"""
        try:
            # 确保目录存在
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error: Failed to save LoFTR config: {e}")
            return False
    
    def get(self, key: str, default=None):
        """获取配置值（支持点号分隔的嵌套键）"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """设置配置值（支持点号分隔的嵌套键）"""
        keys = key.split('.')
        target = self.config
        
        # 导航到目标位置
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # 设置值
        target[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """批量更新配置"""
        self._deep_update(self.config, updates)
    
    def reset_to_defaults(self):
        """重置为默认配置"""
        self.config = self.defaults.copy()
    
    def get_camera_matrix(self):
        """获取相机内参矩阵"""
        import numpy as np
        
        if self.get('camera.use_auto_focal_length', True):
            # 自动计算焦距
            img_width = self.get('camera.img_width', 640)
            img_height = self.get('camera.img_height', 480)
            scale = self.get('camera.focal_length_scale', 1.0)
            
            fx = fy = img_width * scale
            cx = img_width / 2.0
            cy = img_height / 2.0
        else:
            # 使用手动设置的参数
            fx = self.get('camera.manual_fx', 640.0)
            fy = self.get('camera.manual_fy', 640.0)
            cx = self.get('camera.manual_cx', 320.0)
            cy = self.get('camera.manual_cy', 240.0)
        
        return np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]], dtype=np.float32)
    
    def get_loftr_config_dict(self):
        """获取LoFTR模型配置字典"""
        return {
            'MATCH_COARSE': {
                'MATCH_TYPE': self.get('loftr_model.match_type', 'dual_softmax'),
                'SPARSE_SPVS': True,
                'TEMPERATURE': self.get('loftr_model.temperature', 0.1),
                'BORDER_RM': self.get('loftr_model.border_rm', 2),
                'MATCH_THRESHOLD': self.get('loftr_model.match_threshold', 0.2),
                'SKH_ITERS': self.get('loftr_model.skh_iters', 3),
                'SKH_INIT_BIN_SCORE': self.get('loftr_model.skh_init_bin_score', 1.0)
            },
            'COARSE': {
                'D_MODEL': 256,
                'D_FFN': 256,
                'NHEAD': 8,
                'LAYER_NAMES': ['self', 'cross'] * 4,
                'ATTENTION': 'linear',
                'NPE': self.get('loftr_model.npe_config', [832, 832, 832, 832]),
                'RESOLUTION': self.get('loftr_model.coarse_resolution', [8, 2])
            },
            'FINE': {
                'D_MODEL': 128,
                'D_FFN': 128,
                'NHEAD': 8,
                'LAYER_NAMES': ['self', 'cross'] * 1,
                'ATTENTION': 'linear',
                'RESOLUTION': self.get('loftr_model.fine_resolution', 2)
            }
        }
    
    def get_ui_config(self) -> Dict[str, Any]:
        """获取UI显示用的配置"""
        return {
            'enabled': self.get('enabled', True),
            'camera0_id': self.get('camera.camera0_id', 0),
            'camera1_id': self.get('camera.camera1_id', 2),
            'conf_thresh': self.get('matching.conf_thresh', 0.2),
            'resize_factor': self.get('matching.resize_factor', 0.8),
            'skip_frames': self.get('matching.skip_frames', 0),
            'img_width': self.get('camera.img_width', 640),
            'img_height': self.get('camera.img_height', 480),
            'use_auto_focal_length': self.get('camera.use_auto_focal_length', True),
            'focal_length_scale': self.get('camera.focal_length_scale', 1.0),
            'manual_fx': self.get('camera.manual_fx', 640.0),
            'manual_fy': self.get('camera.manual_fy', 640.0),
            'ransac_confidence': self.get('reconstruction.ransac_confidence', 0.999),
            'ransac_threshold': self.get('reconstruction.ransac_threshold', 1.0),
            'depth_threshold': self.get('reconstruction.depth_threshold', 50.0),
            'min_matches': self.get('matching.min_matches', 8),
            'max_matches': self.get('matching.max_matches', 2000),
            'enable_stable_filter': self.get('filtering.enable_stable_filter', True),
            'filter_before_3d': self.get('filtering.filter_before_3d', False)
        }
    
    def apply_ui_config(self, ui_config: Dict[str, Any]):
        """从UI配置更新内部配置"""
        # 映射UI配置到内部配置结构
        mappings = {
            'enabled': 'enabled',
            'camera0_id': 'camera.camera0_id',
            'camera1_id': 'camera.camera1_id',
            'conf_thresh': 'matching.conf_thresh',
            'resize_factor': 'matching.resize_factor',
            'skip_frames': 'matching.skip_frames',
            'img_width': 'camera.img_width',
            'img_height': 'camera.img_height',
            'use_auto_focal_length': 'camera.use_auto_focal_length',
            'focal_length_scale': 'camera.focal_length_scale',
            'manual_fx': 'camera.manual_fx',
            'manual_fy': 'camera.manual_fy',
            'ransac_confidence': 'reconstruction.ransac_confidence',
            'ransac_threshold': 'reconstruction.ransac_threshold',
            'depth_threshold': 'reconstruction.depth_threshold',
            'min_matches': 'matching.min_matches',
            'max_matches': 'matching.max_matches',
            'enable_stable_filter': 'filtering.enable_stable_filter',
            'filter_before_3d': 'filtering.filter_before_3d'
        }
        
        for ui_key, config_key in mappings.items():
            if ui_key in ui_config:
                self.set(config_key, ui_config[ui_key])
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def validate_config(self) -> bool:
        """验证配置的有效性"""
        try:
            # 检查相机ID
            cam0_id = self.get('camera.camera0_id', 0)
            cam1_id = self.get('camera.camera1_id', 2)
            if cam0_id == cam1_id:
                print(f"Warning: Camera IDs are the same: {cam0_id}")
                return False
            
            # 检查置信度阈值
            conf_thresh = self.get('matching.conf_thresh', 0.2)
            if conf_thresh < 0.0 or conf_thresh > 1.0:
                print(f"Warning: Invalid confidence threshold: {conf_thresh}")
                return False
            
            # 检查图像尺寸
            width = self.get('camera.img_width', 640)
            height = self.get('camera.img_height', 480)
            if width <= 0 or height <= 0:
                print(f"Warning: Invalid image size: {width}x{height}")
                return False
            
            # 检查权重文件
            weights_path = Path(self.get('weights_path', ''))
            if not weights_path.exists():
                print(f"Warning: Weights file not found: {weights_path}")
                return False
            
            return True
        except Exception as e:
            print(f"Error validating LoFTR config: {e}")
            return False