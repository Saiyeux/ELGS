#!/usr/bin/env python3
"""
ELGS - 三路匹配系统
实现立体匹配 + 双目时序匹配的增强型特征匹配系统
包含：左右立体匹配、左时序匹配、右时序匹配
"""

import os
import sys
import time
import argparse
from pathlib import Path
from copy import deepcopy
from collections import deque
import threading
import numpy as np

# 避免OpenCV和PyQt冲突
import cv2
cv2.setUseOptimized(True)
os.environ['OPENCV_VIDEOIO_PRIORITY_QT'] = '0'
if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
    del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                                 QHBoxLayout, QWidget, QPushButton, 
                                 QTextEdit, QSplitter, QGroupBox, QComboBox, 
                                 QSpinBox, QFormLayout, QDialog, QDialogButtonBox,
                                 QCheckBox, QTabWidget)
    from PyQt5.QtGui import QPixmap, QImage, QFont
    from PyQt5.QtCore import pyqtSignal, QThread, Qt, QTimer
    QT_AVAILABLE = True
    print("✓ PyQt5 GUI可用")
except ImportError:
    QT_AVAILABLE = False
    print("✗ PyQt5不可用，无法运行GUI界面")
    sys.exit(1)

import torch
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# 添加EfficientLoFTR路径
sys.path.append('thirdparty/EfficientLoFTR')
sys.path.append('thirdparty/EfficientLoFTR/src')

try:
    from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
    print("✓ EfficientLoFTR模块导入成功")
except ImportError as e:
    print(f"✗ EfficientLoFTR模块导入失败: {e}")
    sys.exit(1)


class FrameBuffer:
    """帧缓冲管理器，管理历史帧数据"""
    
    def __init__(self, buffer_size=3):
        self.buffer_size = buffer_size
        self.left_frames = deque(maxlen=buffer_size)
        self.right_frames = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.lock = threading.RLock()
        
    def add_frame_pair(self, left_frame, right_frame, timestamp=None):
        """添加新的帧对"""
        if timestamp is None:
            timestamp = time.time()
            
        with self.lock:
            self.left_frames.append(left_frame.copy())
            self.right_frames.append(right_frame.copy())
            self.timestamps.append(timestamp)
    
    def get_latest_pair(self):
        """获取最新帧对"""
        with self.lock:
            if len(self.left_frames) == 0:
                return None, None, None
            return self.left_frames[-1], self.right_frames[-1], self.timestamps[-1]
    
    def get_previous_pair(self, offset=1):
        """获取历史帧对"""
        with self.lock:
            if len(self.left_frames) <= offset:
                return None, None, None
            idx = -(offset + 1)
            return self.left_frames[idx], self.right_frames[idx], self.timestamps[idx]
    
    def has_temporal_pair(self):
        """检查是否有足够的历史帧进行时序匹配"""
        with self.lock:
            return len(self.left_frames) >= 2
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.left_frames.clear()
            self.right_frames.clear()
            self.timestamps.clear()


class MatchingScheduler:
    """匹配调度器，控制三路匹配的执行策略"""
    
    def __init__(self, strategy='alternate'):
        self.strategy = strategy  # 'all', 'alternate', 'priority'
        self.match_cycle = 0
        self.lock = threading.Lock()
        
        # 策略权重配置
        self.strategy_weights = {
            'stereo': 1.0,      # 立体匹配权重
            'left_temporal': 0.8,   # 左时序匹配权重
            'right_temporal': 0.8   # 右时序匹配权重
        }
    
    def get_next_matching_tasks(self, frame_buffer):
        """根据策略返回下一轮匹配任务"""
        with self.lock:
            self.match_cycle += 1
            
            if self.strategy == 'all':
                # 每帧执行所有匹配
                tasks = ['stereo']
                if frame_buffer.has_temporal_pair():
                    tasks.extend(['left_temporal', 'right_temporal'])
                return tasks
                
            elif self.strategy == 'alternate':
                # 交替执行不同匹配类型
                cycle = self.match_cycle % 3
                if cycle == 1:
                    return ['stereo']
                elif cycle == 2 and frame_buffer.has_temporal_pair():
                    return ['left_temporal']
                elif cycle == 0 and frame_buffer.has_temporal_pair():
                    return ['right_temporal']
                else:
                    return ['stereo']  # 默认执行立体匹配
                    
            elif self.strategy == 'priority':
                # 优先级执行：立体匹配优先
                tasks = ['stereo']
                if self.match_cycle % 2 == 0 and frame_buffer.has_temporal_pair():
                    tasks.append('left_temporal')
                if self.match_cycle % 3 == 0 and frame_buffer.has_temporal_pair():
                    tasks.append('right_temporal')
                return tasks
                
        return ['stereo']  # 默认返回立体匹配


class GPUMemoryManager:
    """GPU内存管理器，优化多匹配器的内存使用"""
    
    def __init__(self):
        self.matchers = {}
        self.matcher_usage = {}
        self.max_concurrent = 2
        self.lock = threading.Lock()
        
    def get_matcher(self, task_type, model_config):
        """获取指定类型的匹配器"""
        with self.lock:
            if task_type not in self.matchers:
                self.matchers[task_type] = self._create_matcher(model_config)
                self.matcher_usage[task_type] = 0
            
            self.matcher_usage[task_type] += 1
            return self.matchers[task_type]
    
    def _create_matcher(self, model_config):
        """创建新的匹配器实例"""
        # 选择配置
        if model_config.get('model_type', 'full') == 'full':
            _default_cfg = deepcopy(full_default_cfg)
        else:
            _default_cfg = deepcopy(opt_default_cfg)
            
        # 精度配置
        precision = model_config.get('precision', 'fp32')
        if precision == 'mp':
            _default_cfg['mp'] = True
        elif precision == 'fp16':
            _default_cfg['half'] = True
            
        # 创建模型
        matcher = LoFTR(config=_default_cfg)
        
        # 加载权重
        weights_path = Path('thirdparty/EfficientLoFTR/weights/eloftr_outdoor.ckpt')
        if not weights_path.exists():
            raise FileNotFoundError("权重文件不存在，请下载eloftr_outdoor.ckpt")
            
        checkpoint = torch.load(str(weights_path), weights_only=False)
        matcher.load_state_dict(checkpoint['state_dict'])
        
        # 重参数化
        matcher = reparameter(matcher)
        
        # 设置设备和精度
        if torch.cuda.is_available():
            if precision == 'fp16':
                matcher = matcher.half()
            matcher = matcher.eval().cuda()
        else:
            matcher = matcher.eval()
            
        return matcher
    
    def release_matcher(self, task_type):
        """释放匹配器使用计数"""
        with self.lock:
            if task_type in self.matcher_usage:
                self.matcher_usage[task_type] = max(0, self.matcher_usage[task_type] - 1)
    
    def cleanup_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_memory_usage(self):
        """获取当前内存使用情况"""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "total": 0}
        
        return {
            "allocated": torch.cuda.memory_allocated() // (1024**2),  # MB
            "reserved": torch.cuda.memory_reserved() // (1024**2),    # MB  
            "total": torch.cuda.get_device_properties(0).total_memory // (1024**2)  # MB
        }


class TripleMatchingResult:
    """三路匹配结果容器"""
    
    def __init__(self):
        self.stereo_matches = None
        self.left_temporal_matches = None
        self.right_temporal_matches = None
        self.match_counts = [0, 0, 0]  # [stereo, left_temporal, right_temporal]
        self.confidences = [[], [], []]
        self.timestamp = time.time()
        
    def add_stereo_result(self, mkpts0, mkpts1, mconf):
        """添加立体匹配结果"""
        self.stereo_matches = {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
            'mconf': mconf
        }
        self.match_counts[0] = len(mkpts0)
        self.confidences[0] = mconf.tolist() if len(mconf) > 0 else []
    
    def add_left_temporal_result(self, mkpts0, mkpts1, mconf):
        """添加左时序匹配结果"""
        self.left_temporal_matches = {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1, 
            'mconf': mconf
        }
        self.match_counts[1] = len(mkpts0)
        self.confidences[1] = mconf.tolist() if len(mconf) > 0 else []
    
    def add_right_temporal_result(self, mkpts0, mkpts1, mconf):
        """添加右时序匹配结果"""
        self.right_temporal_matches = {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
            'mconf': mconf
        }
        self.match_counts[2] = len(mkpts0)
        self.confidences[2] = mconf.tolist() if len(mconf) > 0 else []


class TemporalMatchVisualizer:
    """时序匹配可视化器"""
    
    @staticmethod
    def draw_temporal_matches(img, mkpts0, mkpts1, mconf, conf_thresh=0.2, color=(0, 255, 0), resize_factor=1.0):
        """在图像上绘制时序匹配轨迹"""
        if len(mkpts0) == 0:
            return img
            
        result_img = img.copy()
        high_conf_mask = mconf > conf_thresh
        
        # 计算缩放比例：从预处理图像坐标到原始图像坐标
        scale_factor = 1.0 / resize_factor if resize_factor != 0 else 1.0
        
        for i in range(len(mkpts0)):
            if not high_conf_mask[i]:
                continue
                
            # 将坐标从预处理图像缩放到原始图像
            pt0 = (int(mkpts0[i][0] * scale_factor), int(mkpts0[i][1] * scale_factor))
            pt1 = (int(mkpts1[i][0] * scale_factor), int(mkpts1[i][1] * scale_factor))
            
            # 确保坐标在图像范围内
            h, w = img.shape[:2]
            pt0 = (max(0, min(w-1, pt0[0])), max(0, min(h-1, pt0[1])))
            pt1 = (max(0, min(w-1, pt1[0])), max(0, min(h-1, pt1[1])))
            
            # 绘制轨迹线
            cv2.arrowedLine(result_img, pt0, pt1, color, 2, tipLength=0.3)
            
            # 绘制匹配点
            cv2.circle(result_img, pt0, 3, color, -1)
            cv2.circle(result_img, pt1, 5, color, 2)
            
        return result_img
    
    @staticmethod  
    def draw_stereo_matches(img0, img1, mkpts0, mkpts1, mconf, conf_thresh=0.2, resize_factor=1.0):
        """绘制立体匹配可视化"""
        if len(mkpts0) == 0:
            return np.zeros((max(img0.shape[0], img1.shape[0]), img0.shape[1] + img1.shape[1], 3), dtype=np.uint8)
            
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        combined_h = max(h0, h1)
        combined_w = w0 + w1
        combined_img = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
        
        combined_img[:h0, :w0] = img0
        combined_img[:h1, w0:w0+w1] = img1
        
        # 计算缩放比例：从预处理图像坐标到原始图像坐标
        scale_factor = 1.0 / resize_factor if resize_factor != 0 else 1.0
        
        high_conf_mask = mconf > conf_thresh
        for i in range(len(mkpts0)):
            if not high_conf_mask[i]:
                continue
                
            # 将坐标从预处理图像缩放到原始图像
            pt0 = (int(mkpts0[i][0] * scale_factor), int(mkpts0[i][1] * scale_factor))
            pt1 = (int(mkpts1[i][0] * scale_factor + w0), int(mkpts1[i][1] * scale_factor))
            
            # 确保坐标在图像范围内
            pt0 = (max(0, min(w0-1, pt0[0])), max(0, min(h0-1, pt0[1])))
            pt1 = (max(w0, min(combined_w-1, pt1[0])), max(0, min(h1-1, pt1[1])))
            
            conf = mconf[i]
            if conf > 0.8:
                color = (0, 255, 0)
            elif conf > 0.5:
                color = (0, 255, 255)
            else:
                color = (0, 128, 255)
                
            cv2.circle(combined_img, pt0, 3, color, -1)
            cv2.circle(combined_img, pt1, 3, color, -1)
            cv2.line(combined_img, pt0, pt1, color, 1)
            
        return combined_img


class TripleMatchingWorker(QThread):
    """三路匹配工作线程"""
    
    # 信号定义
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, object)  # left_frame, right_frame, matching_result
    status_update = pyqtSignal(str)
    
    def __init__(self, model_config=None, camera_config=None):
        super().__init__()
        self.running = False
        self.model_config = model_config or {}
        self.camera_config = camera_config or {}
        
        # 组件初始化
        self.frame_buffer = FrameBuffer(buffer_size=3)
        self.scheduler = MatchingScheduler(strategy=self.model_config.get('matching_strategy', 'alternate'))
        self.gpu_manager = GPUMemoryManager()
        
        # 相机对象
        self.cap0 = None
        self.cap1 = None
        
        # 配置参数
        self.conf_thresh = self.model_config.get('conf_thresh', 0.2)
        self.resize_factor = self.model_config.get('resize_factor', 0.8)
        
    def init_cameras(self):
        """初始化双目相机"""
        self.status_update.emit("正在初始化双目相机...")
        
        cam0_id = self.camera_config.get('left_cam_id', 0)
        cam1_id = self.camera_config.get('right_cam_id', 2)
        width = self.camera_config.get('width', 1280)
        height = self.camera_config.get('height', 720)
        fps = self.camera_config.get('fps', 30)
        
        # 查找可用相机
        available_cameras = []
        for cam_id in range(10):
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(cam_id)
                    if cam_id == cam0_id:
                        self.cap0 = cap
                    elif cam_id == cam1_id:
                        self.cap1 = cap
                    elif self.cap0 is None:
                        self.cap0 = cap
                    elif self.cap1 is None:
                        self.cap1 = cap
                    else:
                        cap.release()
                else:
                    cap.release()
            else:
                cap.release()
        
        if len(available_cameras) < 2:
            self.status_update.emit(f"错误: 需要至少2个相机，只找到{len(available_cameras)}个")
            return False
            
        # 确保正确分配相机
        if self.cap0 is None or self.cap1 is None:
            if self.cap0: self.cap0.release()
            if self.cap1: self.cap1.release()
                
            self.cap0 = cv2.VideoCapture(available_cameras[0])
            self.cap1 = cv2.VideoCapture(available_cameras[1])
            
            for cap in [self.cap0, self.cap1]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        self.status_update.emit(f"✓ 双目相机初始化成功: {available_cameras[0]} 和 {available_cameras[1]}")
        return True
    
    def preprocess_image(self, img):
        """预处理图像用于LoFTR推理"""
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
            
        # 缩放
        if self.resize_factor != 1.0:
            new_width = int(img_gray.shape[1] * self.resize_factor)
            new_height = int(img_gray.shape[0] * self.resize_factor)
            img_gray = cv2.resize(img_gray, (new_width, new_height))
            
        # 确保尺寸可被32整除
        h, w = img_gray.shape
        new_h = h // 32 * 32
        new_w = w // 32 * 32
        if new_h != h or new_w != w:
            img_gray = cv2.resize(img_gray, (new_w, new_h))
            
        # 转换为张量
        precision = self.model_config.get('precision', 'fp32')
        if precision == 'fp16':
            img_tensor = torch.from_numpy(img_gray)[None][None].half().cuda() / 255.
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            img_tensor = torch.from_numpy(img_gray)[None][None].to(device).float() / 255.
            
        return img_tensor
    
    def execute_matching(self, task_type, img0_tensor, img1_tensor):
        """执行单个匹配任务"""
        try:
            matcher = self.gpu_manager.get_matcher(task_type, self.model_config)
            
            batch = {'image0': img0_tensor, 'image1': img1_tensor}
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            with torch.no_grad():
                if self.model_config.get('precision') == 'mp':
                    with torch.autocast(enabled=True, device_type='cuda'):
                        matcher(batch)
                else:
                    matcher(batch)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
            
            self.gpu_manager.release_matcher(task_type)
            
            return mkpts0, mkpts1, mconf
            
        except Exception as e:
            self.status_update.emit(f"{task_type}匹配错误: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def run(self):
        """主运行循环"""
        if not self.init_cameras():
            return
            
        self.running = True
        frame_count = 0
        last_time = time.time()
        
        try:
            while self.running:
                # 读取帧
                ret0, frame0 = self.cap0.read()
                ret1, frame1 = self.cap1.read()
                
                if not (ret0 and ret1):
                    self.status_update.emit("警告: 无法读取相机帧")
                    continue
                    
                frame_count += 1
                current_time = time.time()
                
                # 添加到帧缓冲
                self.frame_buffer.add_frame_pair(frame0, frame1, current_time)
                
                # 获取匹配任务
                tasks = self.scheduler.get_next_matching_tasks(self.frame_buffer)
                
                # 创建匹配结果容器
                result = TripleMatchingResult()
                
                # 执行匹配任务
                for task in tasks:
                    if task == 'stereo':
                        # 立体匹配：当前左右帧
                        img0_tensor = self.preprocess_image(frame0)
                        img1_tensor = self.preprocess_image(frame1)
                        mkpts0, mkpts1, mconf = self.execute_matching('stereo', img0_tensor, img1_tensor)
                        
                        # 过滤高置信度匹配
                        high_conf_mask = mconf > self.conf_thresh
                        result.add_stereo_result(
                            mkpts0[high_conf_mask],
                            mkpts1[high_conf_mask], 
                            mconf[high_conf_mask]
                        )
                        
                    elif task == 'left_temporal':
                        # 左时序匹配：左目前后帧
                        prev_left, _, _ = self.frame_buffer.get_previous_pair()
                        if prev_left is not None:
                            img0_tensor = self.preprocess_image(prev_left)
                            img1_tensor = self.preprocess_image(frame0)
                            mkpts0, mkpts1, mconf = self.execute_matching('left_temporal', img0_tensor, img1_tensor)
                            
                            high_conf_mask = mconf > self.conf_thresh
                            result.add_left_temporal_result(
                                mkpts0[high_conf_mask],
                                mkpts1[high_conf_mask],
                                mconf[high_conf_mask]
                            )
                            
                    elif task == 'right_temporal':
                        # 右时序匹配：右目前后帧
                        _, prev_right, _ = self.frame_buffer.get_previous_pair()
                        if prev_right is not None:
                            img0_tensor = self.preprocess_image(prev_right)
                            img1_tensor = self.preprocess_image(frame1)
                            mkpts0, mkpts1, mconf = self.execute_matching('right_temporal', img0_tensor, img1_tensor)
                            
                            high_conf_mask = mconf > self.conf_thresh
                            result.add_right_temporal_result(
                                mkpts0[high_conf_mask],
                                mkpts1[high_conf_mask],
                                mconf[high_conf_mask]
                            )
                
                # 计算FPS
                fps = int(1.0 / (current_time - last_time)) if last_time else 0
                last_time = current_time
                
                # 发送结果
                self.frame_ready.emit(frame0, frame1, result)
                
                # 定期清理GPU内存并监控使用情况
                if frame_count % 50 == 0:
                    memory_usage = self.gpu_manager.get_memory_usage()
                    self.gpu_manager.cleanup_memory()
                    
                    # 如果GPU内存使用率超过80%，发出警告
                    if memory_usage["total"] > 0:
                        usage_percent = memory_usage["reserved"] / memory_usage["total"] * 100
                        if usage_percent > 80:
                            self.status_update.emit(f"警告: GPU内存使用率较高 ({usage_percent:.1f}%)")
                        elif frame_count % 200 == 0:  # 每200帧报告一次内存状态
                            self.status_update.emit(f"GPU内存: {memory_usage['reserved']}/{memory_usage['total']}MB ({usage_percent:.1f}%)")
                
                self.msleep(30)  # 约30fps
                
        except Exception as e:
            self.status_update.emit(f"运行错误: {e}")
        finally:
            self.cleanup()
    
    def stop(self):
        """停止工作线程"""
        self.running = False
        self.quit()
        self.wait()
        
    def cleanup(self):
        """清理资源"""
        if self.cap0:
            self.cap0.release()
        if self.cap1:
            self.cap1.release()
        self.gpu_manager.cleanup_memory()


class TripleMatchingConfigDialog(QDialog):
    """三路匹配配置对话框"""
    
    def __init__(self, current_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("三路匹配系统配置")
        self.setModal(True)
        self.setMinimumSize(400, 350)
        
        self.current_config = current_config
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 创建标签页
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # 相机配置标签页
        camera_tab = QWidget()
        camera_layout = QFormLayout(camera_tab)
        
        self.left_cam_combo = QComboBox()
        self.right_cam_combo = QComboBox()
        for i in range(10):
            self.left_cam_combo.addItem(f"相机 {i}", i)
            self.right_cam_combo.addItem(f"相机 {i}", i)
        
        self.left_cam_combo.setCurrentIndex(self.current_config.get('left_cam_id', 0))
        self.right_cam_combo.setCurrentIndex(self.current_config.get('right_cam_id', 2))
        
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(320, 4096)
        self.width_spinbox.setSingleStep(32)
        self.width_spinbox.setValue(self.current_config.get('width', 1280))
        
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(240, 2160)
        self.height_spinbox.setSingleStep(32)
        self.height_spinbox.setValue(self.current_config.get('height', 720))
        
        camera_layout.addRow("左目相机:", self.left_cam_combo)
        camera_layout.addRow("右目相机:", self.right_cam_combo)
        camera_layout.addRow("宽度:", self.width_spinbox)
        camera_layout.addRow("高度:", self.height_spinbox)
        
        tab_widget.addTab(camera_tab, "相机配置")
        
        # 匹配配置标签页
        matching_tab = QWidget()
        matching_layout = QFormLayout(matching_tab)
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItem("全匹配 (最高质量)", "all")
        self.strategy_combo.addItem("交替匹配 (推荐)", "alternate")
        self.strategy_combo.addItem("优先立体 (最快)", "priority")
        
        current_strategy = self.current_config.get('matching_strategy', 'alternate')
        for i in range(self.strategy_combo.count()):
            if self.strategy_combo.itemData(i) == current_strategy:
                self.strategy_combo.setCurrentIndex(i)
                break
        
        self.model_combo = QComboBox()
        self.model_combo.addItem("完整模型 (高质量)", "full")
        self.model_combo.addItem("优化模型 (高速度)", "opt")
        
        current_model = self.current_config.get('model_type', 'full')
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == current_model:
                self.model_combo.setCurrentIndex(i)
                break
        
        matching_layout.addRow("匹配策略:", self.strategy_combo)
        matching_layout.addRow("模型类型:", self.model_combo)
        
        tab_widget.addTab(matching_tab, "匹配配置")
        
        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_config(self):
        return {
            'left_cam_id': self.left_cam_combo.currentData(),
            'right_cam_id': self.right_cam_combo.currentData(),
            'width': self.width_spinbox.value(),
            'height': self.height_spinbox.value(),
            'matching_strategy': self.strategy_combo.currentData(),
            'model_type': self.model_combo.currentData()
        }


class TripleCameraWindow(QMainWindow):
    """三路匹配系统主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ELGS - 三路匹配系统 (立体+时序)')
        self.setGeometry(100, 100, 1600, 1000)
        
        # 配置参数
        self.model_config = {
            'matching_strategy': 'alternate',
            'model_type': 'full',
            'precision': 'fp32',
            'conf_thresh': 0.2,
            'resize_factor': 0.8
        }
        
        self.camera_config = {
            'left_cam_id': 0,
            'right_cam_id': 2,
            'width': 1280,
            'height': 720,
            'fps': 30
        }
        
        # 显示控制
        self.display_options = {
            'show_left_temporal': True,
            'show_right_temporal': True,
            'show_stereo': True
        }
        
        # 工作线程和统计
        self.matching_worker = None
        self.frame_count = 0
        self.match_history = {'stereo': [], 'left_temporal': [], 'right_temporal': []}
        self.fps_history = []
        
        self.init_ui()
        self.init_update_timer()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧：相机显示和匹配可视化
        display_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(display_splitter, 3)
        
        # 相机显示区域
        camera_widget = self.create_camera_display()
        display_splitter.addWidget(camera_widget)
        
        # 匹配可视化区域
        matching_widget = self.create_matching_display()
        display_splitter.addWidget(matching_widget)
        
        display_splitter.setSizes([400, 600])
        
        # 右侧：控制面板
        control_widget = self.create_control_panel()
        main_layout.addWidget(control_widget, 1)
        
        self.statusBar().showMessage('三路匹配系统准备就绪')
        
    def create_camera_display(self):
        """创建相机显示区域"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 相机显示
        camera_layout = QHBoxLayout()
        
        # 左相机
        left_group = QGroupBox("左目相机")
        left_layout = QVBoxLayout(left_group)
        self.left_cam_label = QLabel()
        self.left_cam_label.setMinimumSize(400, 300)
        self.left_cam_label.setStyleSheet("border: 2px solid #3c3c3c; background-color: #2d2d30;")
        self.left_cam_label.setAlignment(Qt.AlignCenter)
        self.left_cam_label.setText("左目相机\n等待连接...")
        left_layout.addWidget(self.left_cam_label)
        camera_layout.addWidget(left_group)
        
        # 右相机
        right_group = QGroupBox("右目相机")
        right_layout = QVBoxLayout(right_group)
        self.right_cam_label = QLabel()
        self.right_cam_label.setMinimumSize(400, 300)
        self.right_cam_label.setStyleSheet("border: 2px solid #3c3c3c; background-color: #2d2d30;")
        self.right_cam_label.setAlignment(Qt.AlignCenter)
        self.right_cam_label.setText("右目相机\n等待连接...")
        right_layout.addWidget(self.right_cam_label)
        camera_layout.addWidget(right_group)
        
        layout.addLayout(camera_layout)
        
        # 三路匹配统计
        stats_group = QGroupBox("三路匹配统计")
        stats_layout = QHBoxLayout(stats_group)
        
        self.stereo_stats = QLabel("立体匹配: 0")
        self.stereo_stats.setFont(QFont("Arial", 12))
        self.left_temporal_stats = QLabel("左时序: 0")
        self.left_temporal_stats.setFont(QFont("Arial", 12))
        self.right_temporal_stats = QLabel("右时序: 0")
        self.right_temporal_stats.setFont(QFont("Arial", 12))
        self.fps_stats = QLabel("FPS: 0")
        self.fps_stats.setFont(QFont("Arial", 12))
        self.gpu_memory_stats = QLabel("GPU显存: --")
        self.gpu_memory_stats.setFont(QFont("Arial", 12))
        
        stats_layout.addWidget(self.stereo_stats)
        stats_layout.addWidget(self.left_temporal_stats)
        stats_layout.addWidget(self.right_temporal_stats)
        stats_layout.addWidget(self.fps_stats)
        stats_layout.addWidget(self.gpu_memory_stats)
        stats_layout.addStretch()
        
        layout.addWidget(stats_group)
        return widget
        
    def create_matching_display(self):
        """创建匹配可视化区域"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 匹配可视化标签页
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # 立体匹配标签页
        self.stereo_label = QLabel()
        self.stereo_label.setMinimumSize(800, 250)
        self.stereo_label.setStyleSheet("border: 2px solid #3c3c3c; background-color: #2d2d30;")
        self.stereo_label.setAlignment(Qt.AlignCenter)
        self.stereo_label.setText("立体匹配可视化\n等待匹配...")
        tab_widget.addTab(self.stereo_label, "立体匹配")
        
        # 左时序匹配标签页
        self.left_temporal_label = QLabel()
        self.left_temporal_label.setMinimumSize(400, 250)
        self.left_temporal_label.setStyleSheet("border: 2px solid #3c3c3c; background-color: #2d2d30;")
        self.left_temporal_label.setAlignment(Qt.AlignCenter)
        self.left_temporal_label.setText("左目时序匹配\n等待匹配...")
        tab_widget.addTab(self.left_temporal_label, "左时序")
        
        # 右时序匹配标签页
        self.right_temporal_label = QLabel()
        self.right_temporal_label.setMinimumSize(400, 250)
        self.right_temporal_label.setStyleSheet("border: 2px solid #3c3c3c; background-color: #2d2d30;")
        self.right_temporal_label.setAlignment(Qt.AlignCenter)
        self.right_temporal_label.setText("右目时序匹配\n等待匹配...")
        tab_widget.addTab(self.right_temporal_label, "右时序")
        
        return widget
        
    def create_control_panel(self):
        """创建控制面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 控制按钮组
        controls_group = QGroupBox("系统控制")
        controls_layout = QVBoxLayout(controls_group)
        
        self.start_button = QPushButton("开始三路匹配")
        self.start_button.clicked.connect(self.start_matching)
        self.start_button.setMinimumHeight(40)
        controls_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("停止匹配")
        self.stop_button.clicked.connect(self.stop_matching)
        self.stop_button.setEnabled(False)
        self.stop_button.setMinimumHeight(40)
        controls_layout.addWidget(self.stop_button)
        
        self.config_button = QPushButton("配置设置")
        self.config_button.clicked.connect(self.show_config_dialog)
        self.config_button.setMinimumHeight(40)
        controls_layout.addWidget(self.config_button)
        
        layout.addWidget(controls_group)
        
        # 显示选项组
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout(display_group)
        
        self.show_stereo_cb = QCheckBox("显示立体匹配")
        self.show_stereo_cb.setChecked(True)
        display_layout.addWidget(self.show_stereo_cb)
        
        self.show_left_temporal_cb = QCheckBox("显示左时序匹配")
        self.show_left_temporal_cb.setChecked(True)
        display_layout.addWidget(self.show_left_temporal_cb)
        
        self.show_right_temporal_cb = QCheckBox("显示右时序匹配")
        self.show_right_temporal_cb.setChecked(True)
        display_layout.addWidget(self.show_right_temporal_cb)
        
        layout.addWidget(display_group)
        
        # 系统信息
        info_group = QGroupBox("系统信息")
        info_layout = QVBoxLayout(info_group)
        
        self.info_label = QLabel()
        self.info_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        self.update_system_info()
        info_layout.addWidget(self.info_label)
        
        layout.addWidget(info_group)
        
        # 状态日志
        log_group = QGroupBox("状态日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        layout.addStretch()
        return widget
        
    def init_update_timer(self):
        """初始化更新定时器"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_system_info)
        self.update_timer.start(5000)  # 每5秒更新一次
        
    def get_gpu_memory_info(self):
        """获取GPU显存信息"""
        if not torch.cuda.is_available():
            return "不可用", ""
            
        gpu_info = f"可用 ({torch.cuda.get_device_name()})"
        gpu_memory_info = ""
        
        if PYNVML_AVAILABLE:
            try:
                # 不需要每次都初始化，只初始化一次
                if not hasattr(self, '_nvml_initialized'):
                    pynvml.nvmlInit()
                    self._nvml_initialized = True
                    
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_mb = mem_info.used // (1024**2)
                total_mb = mem_info.total // (1024**2)
                
                # 同时获取PyTorch的内存使用情况作为对比
                torch_allocated = torch.cuda.memory_allocated() // (1024**2)
                torch_reserved = torch.cuda.memory_reserved() // (1024**2)
                
                gpu_memory_info = f"\n系统显存: {used_mb}/{total_mb}MB ({used_mb/total_mb*100:.1f}%)\nPyTorch占用: {torch_reserved}MB (已分配: {torch_allocated}MB)"
            except Exception as e:
                gpu_memory_info = f"\n显存信息: 获取失败 ({e})"
        else:
            # 备用方案：使用PyTorch获取部分信息
            try:
                allocated_mb = torch.cuda.memory_allocated() // (1024**2)
                reserved_mb = torch.cuda.memory_reserved() // (1024**2)
                total_mb = torch.cuda.get_device_properties(0).total_memory // (1024**2)
                gpu_memory_info = f"\n显存预留: {reserved_mb}/{total_mb}MB\nPyTorch分配: {allocated_mb}MB"
            except Exception as e:
                gpu_memory_info = f"\n显存信息: 获取失败 ({e})"
                
        return gpu_info, gpu_memory_info
    
    def get_gpu_memory_usage_brief(self):
        """获取简短的GPU显存使用信息用于实时显示"""
        if not torch.cuda.is_available():
            return "不可用"
            
        try:
            if PYNVML_AVAILABLE and hasattr(self, '_nvml_initialized'):
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_mb = mem_info.used // (1024**2)
                total_mb = mem_info.total // (1024**2)
                usage_percent = used_mb / total_mb * 100
                return f"{used_mb}/{total_mb}MB ({usage_percent:.1f}%)"
            else:
                # PyTorch备用方案
                reserved_mb = torch.cuda.memory_reserved() // (1024**2)
                return f"{reserved_mb}MB (PyTorch)"
        except Exception as e:
            return f"获取失败: {str(e)[:20]}"

    def update_system_info(self):
        """更新系统信息显示"""
        gpu_info, gpu_memory_info = self.get_gpu_memory_info()
        
        info_text = f"""匹配策略: {self.model_config['matching_strategy']}
模型类型: {self.model_config['model_type']}
计算精度: {self.model_config['precision']}
置信度阈值: {self.model_config['conf_thresh']}
缩放因子: {self.model_config['resize_factor']}
GPU: {gpu_info}{gpu_memory_info}"""
        
        self.info_label.setText(info_text)
        
    def show_config_dialog(self):
        """显示配置对话框"""
        config = {**self.camera_config, **self.model_config}
        dialog = TripleMatchingConfigDialog(config, self)
        if dialog.exec_() == QDialog.Accepted:
            new_config = dialog.get_config()
            
            # 分离相机配置和模型配置
            camera_keys = ['left_cam_id', 'right_cam_id', 'width', 'height']
            for key in camera_keys:
                if key in new_config:
                    self.camera_config[key] = new_config[key]
                    
            model_keys = ['matching_strategy', 'model_type']
            for key in model_keys:
                if key in new_config:
                    self.model_config[key] = new_config[key]
                    
            self.log_text.append("配置已更新")
            self.update_system_info()
            
    def start_matching(self):
        """开始三路匹配"""
        if self.matching_worker is not None:
            return
            
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        self.matching_worker = TripleMatchingWorker(self.model_config, self.camera_config)
        self.matching_worker.frame_ready.connect(self.update_display)
        self.matching_worker.status_update.connect(self.update_status)
        self.matching_worker.start()
        
        self.log_text.append("开始三路匹配系统...")
        
    def stop_matching(self):
        """停止匹配"""
        if self.matching_worker:
            self.matching_worker.stop()
            self.matching_worker = None
            
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        self.log_text.append("匹配系统已停止")
        
    def update_display(self, left_frame, right_frame, result):
        """更新显示"""
        self.frame_count += 1
        
        # 更新相机显示（带时序匹配叠加）
        left_display = left_frame.copy()
        right_display = right_frame.copy()
        
        # 叠加时序匹配轨迹
        if result.left_temporal_matches and self.show_left_temporal_cb.isChecked():
            left_display = TemporalMatchVisualizer.draw_temporal_matches(
                left_display,
                result.left_temporal_matches['mkpts0'],
                result.left_temporal_matches['mkpts1'],
                result.left_temporal_matches['mconf'],
                self.model_config['conf_thresh'],
                (0, 255, 0),  # 绿色
                self.model_config['resize_factor']
            )
            
        if result.right_temporal_matches and self.show_right_temporal_cb.isChecked():
            right_display = TemporalMatchVisualizer.draw_temporal_matches(
                right_display,
                result.right_temporal_matches['mkpts0'],
                result.right_temporal_matches['mkpts1'],
                result.right_temporal_matches['mconf'],
                self.model_config['conf_thresh'],
                (255, 0, 0),  # 蓝色
                self.model_config['resize_factor']
            )
            
        self.display_frame(self.left_cam_label, left_display)
        self.display_frame(self.right_cam_label, right_display)
        
        # 更新立体匹配可视化
        if result.stereo_matches and self.show_stereo_cb.isChecked():
            stereo_img = TemporalMatchVisualizer.draw_stereo_matches(
                left_frame, right_frame,
                result.stereo_matches['mkpts0'],
                result.stereo_matches['mkpts1'],
                result.stereo_matches['mconf'],
                self.model_config['conf_thresh'],
                self.model_config['resize_factor']
            )
            self.display_frame(self.stereo_label, stereo_img)
            
        # 更新时序匹配可视化（独立显示）
        if result.left_temporal_matches:
            left_temporal_img = TemporalMatchVisualizer.draw_temporal_matches(
                left_frame,
                result.left_temporal_matches['mkpts0'],
                result.left_temporal_matches['mkpts1'], 
                result.left_temporal_matches['mconf'],
                self.model_config['conf_thresh'],
                (0, 255, 0),
                self.model_config['resize_factor']
            )
            self.display_frame(self.left_temporal_label, left_temporal_img)
            
        if result.right_temporal_matches:
            right_temporal_img = TemporalMatchVisualizer.draw_temporal_matches(
                right_frame,
                result.right_temporal_matches['mkpts0'],
                result.right_temporal_matches['mkpts1'],
                result.right_temporal_matches['mconf'],
                self.model_config['conf_thresh'],
                (255, 0, 0),
                self.model_config['resize_factor']
            )
            self.display_frame(self.right_temporal_label, right_temporal_img)
        
        # 更新统计信息
        self.stereo_stats.setText(f"立体匹配: {result.match_counts[0]}")
        self.left_temporal_stats.setText(f"左时序: {result.match_counts[1]}")
        self.right_temporal_stats.setText(f"右时序: {result.match_counts[2]}")
        
        # 更新GPU显存使用率（每帧更新）
        gpu_memory_brief = self.get_gpu_memory_usage_brief()
        self.gpu_memory_stats.setText(f"GPU显存: {gpu_memory_brief}")
        
        # 更新历史统计
        for i, key in enumerate(['stereo', 'left_temporal', 'right_temporal']):
            self.match_history[key].append(result.match_counts[i])
            if len(self.match_history[key]) > 100:
                self.match_history[key].pop(0)
        
        # 每10帧更新一次详细GPU信息（提高刷新频率）
        if self.frame_count % 10 == 0:
            self.update_system_info()
        
    def display_frame(self, label, frame):
        """显示帧到标签"""
        try:
            if len(frame.shape) == 3:
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            else:
                h, w = frame.shape
                qt_image = QImage(frame.data, w, h, w, QImage.Format_Grayscale8)
                
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"显示帧错误: {e}")
            
    def update_status(self, message):
        """更新状态"""
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        self.statusBar().showMessage(message)
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.matching_worker:
            self.matching_worker.stop()
        event.accept()


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='ELGS三路匹配系统')
    parser.add_argument('--cam0', type=int, default=0, help='左目相机ID')
    parser.add_argument('--cam1', type=int, default=2, help='右目相机ID')
    parser.add_argument('--model_type', choices=['full', 'opt'], default='full',
                       help='模型类型')
    parser.add_argument('--precision', choices=['fp32', 'mp', 'fp16'], default='fp32',
                       help='计算精度')
    parser.add_argument('--strategy', choices=['all', 'alternate', 'priority'], default='alternate',
                       help='匹配策略')
    parser.add_argument('--conf_thresh', type=float, default=0.2,
                       help='置信度阈值')
    parser.add_argument('--resize_factor', type=float, default=0.8,
                       help='图像缩放因子')
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        print(f"✓ 使用GPU: {torch.cuda.get_device_name()}")
    else:
        print("! 将使用CPU（性能较慢）")
    
    # 创建应用程序
    app = QApplication(sys.argv)
    
    # 创建主窗口
    window = TripleCameraWindow()
    
    # 更新配置
    window.model_config.update({
        'model_type': args.model_type,
        'precision': args.precision,
        'matching_strategy': args.strategy,
        'conf_thresh': args.conf_thresh,
        'resize_factor': args.resize_factor
    })
    
    window.camera_config.update({
        'left_cam_id': args.cam0,
        'right_cam_id': args.cam1
    })
    
    window.show()
    
    print("✓ 三路匹配系统已启动")
    print("功能说明:")
    print("- 立体匹配：左右目间的空间对应")
    print("- 左时序匹配：左目前后帧的运动跟踪") 
    print("- 右时序匹配：右目前后帧的运动跟踪")
    print("- 时序匹配结果将叠加显示在相机画面上")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()