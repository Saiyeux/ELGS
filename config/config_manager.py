#!/usr/bin/env python3
"""
统一配置管理器
管理所有子系统的配置文件
"""

from .gaussian_config import GaussianConfig
from .loftr_config import LoFTRConfig
from typing import Dict, Any

class ConfigManager:
    """统一配置管理器"""
    
    def __init__(self):
        self.gaussian_config = GaussianConfig()
        self.loftr_config = LoFTRConfig()
    
    def load_all_configs(self):
        """加载所有配置"""
        self.gaussian_config = GaussianConfig()
        self.loftr_config = LoFTRConfig()
    
    def save_all_configs(self) -> bool:
        """保存所有配置"""
        gaussian_ok = self.gaussian_config.save_config()
        loftr_ok = self.loftr_config.save_config()
        return gaussian_ok and loftr_ok
    
    def validate_all_configs(self) -> bool:
        """验证所有配置"""
        gaussian_ok = self.gaussian_config.validate_config()
        loftr_ok = self.loftr_config.validate_config()
        return gaussian_ok and loftr_ok
    
    def reset_all_to_defaults(self):
        """重置所有配置为默认值"""
        self.gaussian_config.reset_to_defaults()
        self.loftr_config.reset_to_defaults()
    
    def get_ui_config_dict(self) -> Dict[str, Any]:
        """获取所有UI配置的统一字典"""
        return {
            'gaussian': self.gaussian_config.get_ui_config(),
            'loftr': self.loftr_config.get_ui_config()
        }
    
    def apply_ui_config_dict(self, ui_config_dict: Dict[str, Any]):
        """从UI统一配置字典更新配置"""
        if 'gaussian' in ui_config_dict:
            self.gaussian_config.apply_ui_config(ui_config_dict['gaussian'])
        
        if 'loftr' in ui_config_dict:
            self.loftr_config.apply_ui_config(ui_config_dict['loftr'])
    
    def export_config_summary(self) -> Dict[str, Any]:
        """导出配置摘要（用于调试）"""
        return {
            'gaussian': {
                'enabled': self.gaussian_config.get('enabled'),
                'iterations': self.gaussian_config.get('reconstruction.iterations'),  # 修正路径
                'pixel_scale': self.gaussian_config.get('pixel_scale'),
                'realtime_enabled': self.gaussian_config.get('realtime_reconstruction.enabled')
            },
            'loftr': {
                'enabled': self.loftr_config.get('enabled'),
                'camera_ids': [
                    self.loftr_config.get('camera.camera0_id'),
                    self.loftr_config.get('camera.camera1_id')
                ],
                'conf_thresh': self.loftr_config.get('matching.conf_thresh'),
                'resize_factor': self.loftr_config.get('matching.resize_factor')
            }
        }