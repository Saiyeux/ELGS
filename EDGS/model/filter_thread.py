#!/usr/bin/env python3
"""
匹配滤波线程模块
用于异步处理特征匹配的噪声过滤
"""

from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import time
from collections import deque

try:
    from .stable_match_filter import StableMatchFilter
    FILTER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Stable match filter not available: {e}")
    FILTER_AVAILABLE = False

class FilterThread(QThread):
    """
    稳定匹配滤波线程
    处理特征匹配的噪声过滤和稳定性检测
    """
    
    # 信号定义
    matches_filtered = pyqtSignal(np.ndarray, np.ndarray, dict)  # 过滤后的匹配点和统计信息
    filter_status_updated = pyqtSignal(dict)  # 滤波器状态更新
    error_occurred = pyqtSignal(str)  # 错误信息
    
    def __init__(self, filter_config=None):
        super().__init__()
        
        # 初始化滤波器
        if FILTER_AVAILABLE:
            if filter_config is None:
                filter_config = {
                    'spatial_clustering': {
                        'enabled': True,
                        'eps': 10.0,
                        'min_samples': 3,
                        'cluster_size_threshold': 5
                    },
                    'temporal_stability': {
                        'enabled': True,
                        'window_size': 5,
                        'position_threshold': 15.0,
                        'stability_ratio_threshold': 0.6
                    },
                    'geometric_validation': {
                        'enabled': True,
                        'distance_threshold': 50.0,
                        'angle_threshold': 30.0,
                        'min_inlier_ratio': 0.7
                    },
                    'statistical_outlier': {
                        'enabled': True,
                        'k_neighbors': 8,
                        'std_multiplier': 2.0
                    }
                }
            
            self.filter = StableMatchFilter(filter_config)
        else:
            self.filter = None
        
        # 线程控制
        self.enabled = False
        self.filter_config = filter_config
        
        # 处理队列
        self.match_queue = deque(maxlen=100)  # 最多保存100帧的匹配数据
        self.processing = False
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'total_input_matches': 0,
            'total_output_matches': 0,
            'average_filter_ratio': 0.0,
            'processing_time_ms': 0.0,
            'last_filter_info': {}
        }
        
    def is_available(self):
        """检查滤波器是否可用"""
        return FILTER_AVAILABLE and self.filter is not None
    
    def set_enabled(self, enabled):
        """启用/禁用匹配滤波"""
        self.enabled = enabled
        if enabled and not self.is_available():
            print("Warning: Stable match filter is not available")
            return False
        return True
    
    def update_filter_config(self, config):
        """更新滤波器配置"""
        if not self.is_available():
            return False
        
        try:
            self.filter_config = config
            self.filter.update_config(config)
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to update filter config: {e}")
            return False
    
    def add_matches(self, pts0, pts1, confidence_scores=None):
        """添加新的匹配点对到处理队列"""
        if not self.enabled or not self.is_available():
            # 如果滤波器未启用，直接返回原始匹配
            empty_stats = {
                'filtered': False,
                'input_count': len(pts0),
                'output_count': len(pts0),
                'filter_ratio': 1.0,
                'processing_time_ms': 0.0,
                'filter_info': {}
            }
            self.matches_filtered.emit(pts0, pts1, empty_stats)
            return
        
        if len(pts0) == 0 or len(pts1) == 0:
            empty_stats = {
                'filtered': True,
                'input_count': 0,
                'output_count': 0,
                'filter_ratio': 0.0,
                'processing_time_ms': 0.0,
                'filter_info': {}
            }
            self.matches_filtered.emit(np.array([]), np.array([]), empty_stats)
            return
        
        # 添加到处理队列
        match_data = {
            'pts0': pts0.copy(),
            'pts1': pts1.copy(),
            'confidence_scores': confidence_scores.copy() if confidence_scores is not None else None,
            'timestamp': time.time()
        }
        
        self.match_queue.append(match_data)
        
        # 启动处理（如果没有在处理中）
        if not self.processing and not self.isRunning():
            self.start()
    
    def run(self):
        """线程主循环 - 处理匹配点滤波"""
        self.processing = True
        
        try:
            while len(self.match_queue) > 0 and self.enabled:
                # 获取最新的匹配数据
                match_data = self.match_queue.popleft()
                
                # 执行滤波
                self.process_matches(
                    match_data['pts0'],
                    match_data['pts1'],
                    match_data['confidence_scores']
                )
                
                # 短暂休眠避免CPU占用过高
                self.msleep(10)
                
        except Exception as e:
            self.error_occurred.emit(f"Filter processing error: {e}")
        finally:
            self.processing = False
    
    def process_matches(self, pts0, pts1, confidence_scores=None):
        """处理单帧匹配点"""
        if not self.is_available():
            return
        
        start_time = time.time()
        input_count = len(pts0)
        
        try:
            # 执行滤波
            filtered_pts0, filtered_pts1, filter_info = self.filter.filter_matches(
                pts0, pts1, confidence_scores
            )
            
            # 计算处理时间
            processing_time_ms = (time.time() - start_time) * 1000
            
            # 计算统计信息
            output_count = len(filtered_pts0)
            filter_ratio = output_count / input_count if input_count > 0 else 0.0
            
            # 更新统计
            self.update_statistics(input_count, output_count, filter_ratio, processing_time_ms, filter_info)
            
            # 准备结果统计信息
            result_stats = {
                'filtered': True,
                'input_count': input_count,
                'output_count': output_count,
                'filter_ratio': filter_ratio,
                'processing_time_ms': processing_time_ms,
                'filter_info': filter_info
            }
            
            # 发送过滤结果
            self.matches_filtered.emit(filtered_pts0, filtered_pts1, result_stats)
            
            # 发送状态更新
            self.emit_status_update()
            
        except Exception as e:
            # 滤波失败，返回原始匹配
            error_msg = f"Match filtering failed: {e}"
            self.error_occurred.emit(error_msg)
            
            fallback_stats = {
                'filtered': False,
                'input_count': input_count,
                'output_count': input_count,
                'filter_ratio': 1.0,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'filter_info': {'error': str(e)}
            }
            
            self.matches_filtered.emit(pts0, pts1, fallback_stats)
    
    def update_statistics(self, input_count, output_count, filter_ratio, processing_time_ms, filter_info):
        """更新统计信息"""
        self.stats['total_processed'] += 1
        self.stats['total_input_matches'] += input_count
        self.stats['total_output_matches'] += output_count
        
        # 计算平均滤波比率
        if self.stats['total_input_matches'] > 0:
            self.stats['average_filter_ratio'] = self.stats['total_output_matches'] / self.stats['total_input_matches']
        
        # 更新平均处理时间（移动平均）
        alpha = 0.1  # 平滑因子
        self.stats['processing_time_ms'] = (
            alpha * processing_time_ms + 
            (1 - alpha) * self.stats['processing_time_ms']
        )
        
        self.stats['last_filter_info'] = filter_info
    
    def emit_status_update(self):
        """发送状态更新信号"""
        status = {
            'enabled': self.enabled,
            'available': self.is_available(),
            'processing': self.processing,
            'queue_size': len(self.match_queue),
            'stats': self.stats.copy(),
            'config': self.filter_config
        }
        self.filter_status_updated.emit(status)
    
    def clear_statistics(self):
        """清空统计信息"""
        self.stats = {
            'total_processed': 0,
            'total_input_matches': 0,
            'total_output_matches': 0,
            'average_filter_ratio': 0.0,
            'processing_time_ms': 0.0,
            'last_filter_info': {}
        }
        self.emit_status_update()
    
    def clear_queue(self):
        """清空处理队列"""
        self.match_queue.clear()
    
    def get_filter_methods(self):
        """获取可用的滤波方法列表"""
        if not self.is_available():
            return []
        
        return [
            'spatial_clustering',
            'temporal_stability', 
            'geometric_validation',
            'statistical_outlier'
        ]
    
    def set_filter_method_enabled(self, method, enabled):
        """启用/禁用特定滤波方法"""
        if not self.is_available() or method not in self.filter_config:
            return False
        
        self.filter_config[method]['enabled'] = enabled
        return self.update_filter_config(self.filter_config)
    
    def get_status_info(self):
        """获取当前状态信息（用于UI显示）"""
        if not self.is_available():
            return {
                'available': False,
                'enabled': False,
                'processing': False,
                'queue_size': 0,
                'total_processed': 0,
                'average_filter_ratio': 0.0
            }
        
        return {
            'available': True,
            'enabled': self.enabled,
            'processing': self.processing,
            'queue_size': len(self.match_queue),
            'total_processed': self.stats['total_processed'],
            'average_filter_ratio': self.stats['average_filter_ratio'],
            'processing_time_ms': self.stats['processing_time_ms'],
            'filter_methods': {
                method: config.get('enabled', False) 
                for method, config in self.filter_config.items()
            }
        }