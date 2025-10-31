#!/usr/bin/env python3
"""
3D Gaussian Splatting 配置管理
存储和管理3DGS训练相关的所有参数
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

class GaussianConfig:
    """3DGS配置管理类"""
    
    def __init__(self, config_file="config/gaussian_params.json"):
        self.config_file = Path(config_file)
        
        # 默认配置参数
        self.defaults = {
            # 基本参数
            "enabled": False,
            "output_dir": "output_3dgs",
            "pixel_scale": 0.001,
            "auto_train_threshold": 500,
            
            # 实时重建参数（快速迭代）
            "reconstruction": {
                "iterations": 200,  # 快速重建迭代次数 (1-1000)
                "position_lr_init": 0.001,  # 更高的学习率用于快速收敛
                "position_lr_final": 0.0001,
                "position_lr_delay_mult": 0.01,
                "feature_lr": 0.01,  # 更高的特征学习率
                "opacity_lr": 0.1,   # 更高的不透明度学习率
                "scaling_lr": 0.01,
                "rotation_lr": 0.005,
                "lambda_dssim": 0.1,  # 降低DSSIM权重提高速度
                "percent_dense": 0.1  # 更高的密度
            },
            
            # 密度控制参数（针对快速重建优化）
            "densification": {
                "densify_from_iter_ratio": 0.25,  # 相对于总迭代数的比例
                "densify_until_iter_ratio": 0.5,  # 相对于总迭代数的比例
                "densify_grad_threshold": 0.001,  # 更宽松的阈值
                "densification_interval_ratio": 0.05,  # 相对于总迭代数的比例
                "opacity_reset_interval_ratio": 0.5,  # 相对于总迭代数的比例
                "size_threshold": 10
            },
            
            # 曝光参数
            "exposure": {
                "exposure_lr_init": 0.001,
                "exposure_lr_final": 0.0001,
                "exposure_lr_delay_steps": 5000,
                "exposure_lr_delay_mult": 0.001
            },
            
            # 虚拟相机参数（快速重建优化）
            "virtual_cameras": {
                "num_cameras": 3,  # 减少相机数量提高速度
                "radius_scale": 1.5,
                "fov": 45,
                "width": 200,  # 减小分辨率提高速度
                "height": 150,
                "elevation_angle": 30  # 度
            },
            
            # 实时重建参数
            "realtime_reconstruction": {
                "enabled": True,
                "max_queue_size": 1,  # 只保留最新点云
                "min_points": 10,  # 最少点数才进行重建
                "density_enhancement": True,  # 启用密度增强
                "enhancement_factor": 4  # 密度增强倍数
            },
            
            # 渲染参数
            "rendering": {
                "bg_color": [0, 0, 0],  # RGB
                "antialiasing": False,
                "compute_cov3D_python": False,
                "convert_SHs_python": False,
                "debug": False
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
                print(f"Warning: Failed to load Gaussian config: {e}")
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
            print(f"Error: Failed to save Gaussian config: {e}")
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
    
    def get_training_args_dict(self) -> Dict[str, Any]:
        """获取训练参数字典（用于训练线程）"""
        iterations = self.get('training.iterations', 5000)
        
        return {
            'iterations': iterations,
            'position_lr_init': self.get('training.position_lr_init', 0.00016),
            'position_lr_final': self.get('training.position_lr_final', 0.0000016),
            'position_lr_delay_mult': self.get('training.position_lr_delay_mult', 0.01),
            'position_lr_max_steps': iterations,
            'feature_lr': self.get('training.feature_lr', 0.0025),
            'opacity_lr': self.get('training.opacity_lr', 0.05),
            'scaling_lr': self.get('training.scaling_lr', 0.005),
            'rotation_lr': self.get('training.rotation_lr', 0.001),
            'densify_from_iter': self.get('densification.densify_from_iter', 500),
            'densify_until_iter': min(15000, int(iterations * self.get('densification.densify_until_iter_ratio', 0.5))),
            'densify_grad_threshold': self.get('densification.densify_grad_threshold', 0.0002),
            'densification_interval': self.get('densification.densification_interval', 100),
            'opacity_reset_interval': self.get('densification.opacity_reset_interval', 3000),
            'lambda_dssim': self.get('training.lambda_dssim', 0.2),
            'percent_dense': self.get('training.percent_dense', 0.01),
            'exposure_lr_init': self.get('exposure.exposure_lr_init', 0.001),
            'exposure_lr_final': self.get('exposure.exposure_lr_final', 0.0001),
            'exposure_lr_delay_steps': self.get('exposure.exposure_lr_delay_steps', 5000),
            'exposure_lr_delay_mult': self.get('exposure.exposure_lr_delay_mult', 0.001)
        }
    
    def get_pipeline_args_dict(self) -> Dict[str, Any]:
        """获取渲染管道参数字典"""
        return {
            'convert_SHs_python': self.get('rendering.convert_SHs_python', False),
            'compute_cov3D_python': self.get('rendering.compute_cov3D_python', False),
            'debug': self.get('rendering.debug', False),
            'antialiasing': self.get('rendering.antialiasing', False)
        }
    
    def get_ui_config(self) -> Dict[str, Any]:
        """获取UI显示用的配置"""
        return {
            'enabled': self.get('enabled', False),
            'realtime_enabled': self.get('realtime_reconstruction.enabled', True),
            'iterations': self.get('reconstruction.iterations', 200),
            'pixel_scale': self.get('pixel_scale', 0.001),
            'num_cameras': self.get('virtual_cameras.num_cameras', 3),
            'camera_fov': self.get('virtual_cameras.fov', 45),
            'densify_grad_threshold': self.get('densification.densify_grad_threshold', 0.001),
            'position_lr_init': self.get('reconstruction.position_lr_init', 0.001),
            'feature_lr': self.get('reconstruction.feature_lr', 0.01),
            'lambda_dssim': self.get('reconstruction.lambda_dssim', 0.1),
            'density_enhancement': self.get('realtime_reconstruction.density_enhancement', True),
            'enhancement_factor': self.get('realtime_reconstruction.enhancement_factor', 4),
            'min_points': self.get('realtime_reconstruction.min_points', 10)
        }
    
    def apply_ui_config(self, ui_config: Dict[str, Any]):
        """从UI配置更新内部配置"""
        # 映射UI配置到内部配置结构
        mappings = {
            'enabled': 'enabled',
            'realtime_enabled': 'realtime_reconstruction.enabled',
            'iterations': 'reconstruction.iterations',
            'pixel_scale': 'pixel_scale',
            'num_cameras': 'virtual_cameras.num_cameras',
            'camera_fov': 'virtual_cameras.fov',
            'densify_grad_threshold': 'densification.densify_grad_threshold',
            'position_lr_init': 'reconstruction.position_lr_init',
            'feature_lr': 'reconstruction.feature_lr',
            'lambda_dssim': 'reconstruction.lambda_dssim',
            'density_enhancement': 'realtime_reconstruction.density_enhancement',
            'enhancement_factor': 'realtime_reconstruction.enhancement_factor',
            'min_points': 'realtime_reconstruction.min_points'
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
            # 检查重建迭代次数范围 (1-1000)
            iterations = self.get('reconstruction.iterations', 0)
            if iterations < 1 or iterations > 1000:
                print(f"Warning: Invalid reconstruction iterations value: {iterations} (should be 1-1000)")
                return False
            
            pixel_scale = self.get('pixel_scale', 0)
            if pixel_scale <= 0 or pixel_scale > 1:
                print(f"Warning: Invalid pixel_scale value: {pixel_scale}")
                return False
            
            min_points = self.get('realtime_reconstruction.min_points', 0)
            if min_points < 3 or min_points > 1000:
                print(f"Warning: Invalid min_points value: {min_points}")
                return False
            
            enhancement_factor = self.get('realtime_reconstruction.enhancement_factor', 0)
            if enhancement_factor < 1 or enhancement_factor > 20:
                print(f"Warning: Invalid enhancement_factor value: {enhancement_factor}")
                return False
            
            return True
        except Exception as e:
            print(f"Error validating Gaussian config: {e}")
            return False