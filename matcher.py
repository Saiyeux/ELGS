#!/usr/bin/env python3
"""
双目相机实时特征匹配脚本
整合PyQt5窗口显示和EfficientLoFTR实时匹配功能
参考run.py的窗口创建和thirdparty/EfficientLoFTR的模型配置
"""

import os
import sys
import time
import argparse
from pathlib import Path
from copy import deepcopy

# 完全避免OpenCV的Qt插件冲突
import cv2
# 在导入PyQt5之前先导入OpenCV，并强制禁用Qt后端
cv2.setUseOptimized(True)
# 设置OpenCV不使用Qt
os.environ['OPENCV_VIDEOIO_PRIORITY_QT'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
# 清理Qt插件路径避免冲突  
if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
    del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                                 QHBoxLayout, QWidget, QPushButton, QProgressBar, 
                                 QTextEdit, QSplitter, QGroupBox, QComboBox, 
                                 QSpinBox, QFormLayout, QDialog, QDialogButtonBox)
    from PyQt5.QtGui import QPixmap, QImage, QFont
    from PyQt5.QtCore import pyqtSignal, QThread, Qt
    QT_AVAILABLE = True
    print("✓ PyQt5 GUI可用")
except ImportError:
    QT_AVAILABLE = False
    print("✗ PyQt5不可用，无法运行GUI界面")
    sys.exit(1)

import torch
import numpy as np

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

class CameraWorker(QThread):
    """相机捕获工作线程"""
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, int, int, object)  # frame0, frame1, matches, fps, match_img
    status_update = pyqtSignal(str)
    
    def __init__(self, model_config=None):
        super().__init__()
        self.running = False
        self.model_config = model_config or {}
        
        # 从配置中获取相机参数
        self.cam0_id = self.model_config.get('left_cam_id', 0)
        self.cam1_id = self.model_config.get('right_cam_id', 2)
        self.cam_width = self.model_config.get('width', 1280)
        self.cam_height = self.model_config.get('height', 720)
        self.cam_fps = self.model_config.get('fps', 30)
        
        # 相机对象
        self.cap0 = None
        self.cap1 = None
        
        # EfficientLoFTR匹配器
        self.matcher = None
        
        # 配置参数
        self.conf_thresh = self.model_config.get('conf_thresh', 0.2)
        self.resize_factor = self.model_config.get('resize_factor', 0.8)
        self.model_type = self.model_config.get('model_type', 'full')
        self.precision = self.model_config.get('precision', 'fp32')
        
    def init_cameras(self):
        """初始化相机"""
        self.status_update.emit("正在初始化相机...")
        
        # 查找可用相机
        available_cameras = []
        for cam_id in range(10):
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                # 设置相机参数
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
                cap.set(cv2.CAP_PROP_FPS, self.cam_fps)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                
                # 测试读取
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(cam_id)
                    if len(available_cameras) >= 2:
                        if len(available_cameras) == 1:
                            self.cap0 = cap
                        else:
                            if cam_id == self.cam0_id:
                                self.cap0 = cap
                            elif cam_id == self.cam1_id:
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
        
        # 确保有两个相机
        if len(available_cameras) < 2:
            self.status_update.emit(f"错误: 需要至少2个相机，只找到{len(available_cameras)}个")
            return False
            
        # 如果没有正确分配相机，重新分配
        if self.cap0 is None or self.cap1 is None:
            # 释放所有已打开的相机
            if self.cap0:
                self.cap0.release()
            if self.cap1:
                self.cap1.release()
                
            # 重新打开前两个可用相机
            self.cap0 = cv2.VideoCapture(available_cameras[0])
            self.cap1 = cv2.VideoCapture(available_cameras[1])
            
            # 重新设置参数
            for cap in [self.cap0, self.cap1]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
                cap.set(cv2.CAP_PROP_FPS, self.cam_fps)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        self.status_update.emit(f"✓ 相机初始化成功: {available_cameras[0]} 和 {available_cameras[1]}")
        return True
        
    def init_matcher(self):
        """初始化EfficientLoFTR匹配器"""
        self.status_update.emit("正在初始化EfficientLoFTR匹配器...")
        
        try:
            # 选择配置
            if self.model_type == 'full':
                _default_cfg = deepcopy(full_default_cfg)
            elif self.model_type == 'opt':
                _default_cfg = deepcopy(opt_default_cfg)
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
                
            # 精度配置
            if self.precision == 'mp':
                _default_cfg['mp'] = True
            elif self.precision == 'fp16':
                _default_cfg['half'] = True
                
            # 创建模型
            self.matcher = LoFTR(config=_default_cfg)
            
            # 加载权重
            weights_path = Path('thirdparty/EfficientLoFTR/weights/eloftr_outdoor.ckpt')
            if not weights_path.exists():
                self.status_update.emit("✗ 权重文件不存在，请下载eloftr_outdoor.ckpt")
                return False
                
            checkpoint = torch.load(str(weights_path), weights_only=False)
            self.matcher.load_state_dict(checkpoint['state_dict'])
            
            # 重参数化（必须）
            self.matcher = reparameter(self.matcher)
            
            # 设置设备和精度
            if torch.cuda.is_available():
                if self.precision == 'fp16':
                    self.matcher = self.matcher.half()
                self.matcher = self.matcher.eval().cuda()
                self.status_update.emit("✓ 使用GPU进行匹配")
            else:
                self.matcher = self.matcher.eval()
                self.status_update.emit("! 使用CPU进行匹配（性能较慢）")
                
            self.status_update.emit("✓ EfficientLoFTR匹配器初始化成功")
            return True
            
        except Exception as e:
            self.status_update.emit(f"✗ 匹配器初始化失败: {e}")
            return False
    
    def preprocess_image(self, img):
        """预处理图像用于LoFTR推理"""
        # 转换为灰度图
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
            
        # 缩放图像
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
        if self.precision == 'fp16':
            img_tensor = torch.from_numpy(img_gray)[None][None].half().cuda() / 255.
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            img_tensor = torch.from_numpy(img_gray)[None][None].to(device).float() / 255.
            
        return img_gray, img_tensor
    
    def draw_matches(self, img0, img1, mkpts0, mkpts1, mconf, conf_thresh=0.2):
        """在图像上绘制匹配点和连接线"""
        # 过滤高置信度匹配
        high_conf_mask = mconf > conf_thresh
        mkpts0_filtered = mkpts0[high_conf_mask]
        mkpts1_filtered = mkpts1[high_conf_mask]
        mconf_filtered = mconf[high_conf_mask]
        
        # 创建匹配可视化图像
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        
        # 创建合并图像（水平拼接）
        combined_h = max(h0, h1)
        combined_w = w0 + w1
        combined_img = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
        
        # 将原图像复制到合并图像中
        combined_img[:h0, :w0] = img0
        combined_img[:h1, w0:w0+w1] = img1
        
        # 绘制匹配点和连接线
        for i in range(len(mkpts0_filtered)):
            # 缩放匹配点坐标（考虑图像预处理时的缩放）
            pt0 = (int(mkpts0_filtered[i][0] / self.resize_factor), 
                   int(mkpts0_filtered[i][1] / self.resize_factor))
            pt1 = (int(mkpts1_filtered[i][0] / self.resize_factor + w0), 
                   int(mkpts1_filtered[i][1] / self.resize_factor))
            
            # 根据置信度设置颜色
            conf = mconf_filtered[i]
            if conf > 0.8:
                color = (0, 255, 0)  # 绿色 - 高置信度
            elif conf > 0.5:
                color = (0, 255, 255)  # 黄色 - 中等置信度
            else:
                color = (0, 128, 255)  # 橙色 - 较低置信度
            
            # 绘制匹配点
            cv2.circle(combined_img, pt0, 3, color, -1)
            cv2.circle(combined_img, pt1, 3, color, -1)
            
            # 绘制连接线
            cv2.line(combined_img, pt0, pt1, color, 1)
        
        return combined_img
    
    def run(self):
        """主运行循环"""
        # 初始化
        if not self.init_cameras():
            return
            
        if not self.init_matcher():
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
                
                # 执行匹配
                num_matches = 0
                match_img = None
                try:
                    if self.matcher is not None:
                        # 预处理图像
                        img0_gray, img0_tensor = self.preprocess_image(frame0)
                        img1_gray, img1_tensor = self.preprocess_image(frame1)
                        
                        # 执行匹配
                        batch = {'image0': img0_tensor, 'image1': img1_tensor}
                        
                        # 强制同步GPU以确保显存状态更新
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            
                        with torch.no_grad():
                            if self.precision == 'mp':
                                with torch.autocast(enabled=True, device_type='cuda'):
                                    self.matcher(batch)
                            else:
                                self.matcher(batch)
                        
                        # 推理完成后同步GPU状态
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        # 获取匹配点
                        mkpts0 = batch['mkpts0_f'].cpu().numpy()
                        mkpts1 = batch['mkpts1_f'].cpu().numpy()
                        mconf = batch['mconf'].cpu().numpy()
                        
                        # 过滤高置信度匹配
                        high_conf_mask = mconf > self.conf_thresh
                        num_matches = np.sum(high_conf_mask)
                        
                        # 绘制匹配可视化
                        if num_matches > 0:
                            match_img = self.draw_matches(frame0, frame1, mkpts0, mkpts1, mconf, self.conf_thresh)
                        
                except Exception as e:
                    self.status_update.emit(f"匹配错误: {e}")
                
                # 计算FPS
                current_time = time.time()
                fps = int(1.0 / (current_time - last_time)) if last_time else 0
                last_time = current_time
                
                # 发送帧数据（包括匹配可视化图像）
                self.frame_ready.emit(frame0, frame1, num_matches, fps, match_img)
                
                # 短暂休眠
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


class CameraConfigDialog(QDialog):
    """相机配置对话框"""
    
    def __init__(self, current_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("相机配置")
        self.setModal(True)
        self.setMinimumSize(300, 250)
        
        self.current_config = current_config
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建表单布局
        form_layout = QFormLayout()
        
        # 左目相机ID
        self.left_cam_combo = QComboBox()
        for i in range(10):
            self.left_cam_combo.addItem(f"相机 {i}", i)
        self.left_cam_combo.setCurrentIndex(self.current_config['left_cam_id'])
        form_layout.addRow("左目相机:", self.left_cam_combo)
        
        # 右目相机ID
        self.right_cam_combo = QComboBox()
        for i in range(10):
            self.right_cam_combo.addItem(f"相机 {i}", i)
        self.right_cam_combo.setCurrentIndex(self.current_config['right_cam_id'])
        form_layout.addRow("右目相机:", self.right_cam_combo)
        
        # 分辨率宽度
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(320, 4096)
        self.width_spinbox.setSingleStep(32)
        self.width_spinbox.setValue(self.current_config['width'])
        form_layout.addRow("宽度:", self.width_spinbox)
        
        # 分辨率高度
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(240, 2160)
        self.height_spinbox.setSingleStep(32)
        self.height_spinbox.setValue(self.current_config['height'])
        form_layout.addRow("高度:", self.height_spinbox)
        
        # 帧率
        self.fps_combo = QComboBox()
        fps_options = [15, 30, 60, 120]
        for fps in fps_options:
            self.fps_combo.addItem(f"{fps} FPS", fps)
        current_fps_index = fps_options.index(self.current_config['fps']) if self.current_config['fps'] in fps_options else 1
        self.fps_combo.setCurrentIndex(current_fps_index)
        form_layout.addRow("帧率:", self.fps_combo)
        
        layout.addLayout(form_layout)
        
        # 预设分辨率按钮
        preset_layout = QHBoxLayout()
        
        preset_720p = QPushButton("720P")
        preset_720p.clicked.connect(lambda: self.set_resolution(1280, 720))
        preset_layout.addWidget(preset_720p)
        
        preset_1080p = QPushButton("1080P")
        preset_1080p.clicked.connect(lambda: self.set_resolution(1920, 1080))
        preset_layout.addWidget(preset_1080p)
        
        preset_480p = QPushButton("480P")
        preset_480p.clicked.connect(lambda: self.set_resolution(640, 480))
        preset_layout.addWidget(preset_480p)
        
        layout.addLayout(preset_layout)
        
        # 对话框按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def set_resolution(self, width, height):
        """设置预设分辨率"""
        self.width_spinbox.setValue(width)
        self.height_spinbox.setValue(height)
        
    def get_config(self):
        """获取配置"""
        return {
            'left_cam_id': self.left_cam_combo.currentData(),
            'right_cam_id': self.right_cam_combo.currentData(),
            'width': self.width_spinbox.value(),
            'height': self.height_spinbox.value(),
            'fps': self.fps_combo.currentData()
        }


class DualCameraWindow(QMainWindow):
    """双目相机主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ELGS - 双目相机实时特征匹配')
        self.setGeometry(100, 100, 1400, 900)
        
        # 配置参数
        self.model_config = {
            'conf_thresh': 0.2,
            'resize_factor': 0.8,
            'model_type': 'full',
            'precision': 'fp32'
        }
        
        # 相机配置
        self.camera_config = {
            'left_cam_id': 0,
            'right_cam_id': 2,
            'width': 1280,
            'height': 720,
            'fps': 30
        }
        
        # 工作线程
        self.camera_worker = None
        
        # 统计信息
        self.frame_count = 0
        self.match_history = []
        self.fps_history = []
        
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧相机显示区域
        camera_widget = self.create_camera_display()
        splitter.addWidget(camera_widget)
        
        # 右侧控制和状态区域
        control_widget = self.create_control_panel()
        splitter.addWidget(control_widget)
        
        # 设置分割比例
        splitter.setSizes([1000, 400])
        
        # 状态栏
        self.statusBar().showMessage('准备启动双目相机匹配...')
        
    def create_camera_display(self):
        """创建相机显示区域"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 相机显示区域
        camera_layout = QHBoxLayout()
        
        # 左相机
        cam0_group = QGroupBox("相机 0")
        cam0_layout = QVBoxLayout(cam0_group)
        self.cam0_label = QLabel()
        self.cam0_label.setMinimumSize(480, 360)
        self.cam0_label.setStyleSheet("border: 2px solid #3c3c3c; background-color: #2d2d30;")
        self.cam0_label.setAlignment(Qt.AlignCenter)
        self.cam0_label.setText("相机 0\n等待连接...")
        cam0_layout.addWidget(self.cam0_label)
        camera_layout.addWidget(cam0_group)
        
        # 右相机
        cam1_group = QGroupBox("相机 1")
        cam1_layout = QVBoxLayout(cam1_group)
        self.cam1_label = QLabel()
        self.cam1_label.setMinimumSize(480, 360)
        self.cam1_label.setStyleSheet("border: 2px solid #3c3c3c; background-color: #2d2d30;")
        self.cam1_label.setAlignment(Qt.AlignCenter)
        self.cam1_label.setText("相机 1\n等待连接...")
        cam1_layout.addWidget(self.cam1_label)
        camera_layout.addWidget(cam1_group)
        
        layout.addLayout(camera_layout)
        
        # 匹配可视化区域
        match_group = QGroupBox("匹配可视化")
        match_layout = QVBoxLayout(match_group)
        self.match_label = QLabel()
        self.match_label.setMinimumSize(800, 300)
        self.match_label.setStyleSheet("border: 2px solid #3c3c3c; background-color: #2d2d30;")
        self.match_label.setAlignment(Qt.AlignCenter)
        self.match_label.setText("匹配可视化\n等待匹配...")
        match_layout.addWidget(self.match_label)
        layout.addWidget(match_group)
        
        # 匹配统计信息
        stats_group = QGroupBox("匹配统计")
        stats_layout = QHBoxLayout(stats_group)
        
        self.matches_label = QLabel("匹配点数: 0")
        self.matches_label.setFont(QFont("Arial", 14))
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setFont(QFont("Arial", 14))
        self.avg_matches_label = QLabel("平均匹配: 0")
        self.avg_fps_label = QLabel("平均FPS: 0")
        
        stats_layout.addWidget(self.matches_label)
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.avg_matches_label)
        stats_layout.addWidget(self.avg_fps_label)
        stats_layout.addStretch()
        
        layout.addWidget(stats_group)
        
        return widget
        
    def create_control_panel(self):
        """创建控制面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 控制按钮
        controls_group = QGroupBox("控制")
        controls_layout = QVBoxLayout(controls_group)
        
        self.start_button = QPushButton("开始匹配")
        self.start_button.clicked.connect(self.start_matching)
        self.start_button.setMinimumHeight(40)
        controls_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("停止匹配")
        self.stop_button.clicked.connect(self.stop_matching)
        self.stop_button.setEnabled(False)
        self.stop_button.setMinimumHeight(40)
        controls_layout.addWidget(self.stop_button)
        
        self.reset_button = QPushButton("重置统计")
        self.reset_button.clicked.connect(self.reset_stats)
        self.reset_button.setMinimumHeight(40)
        controls_layout.addWidget(self.reset_button)
        
        self.config_button = QPushButton("相机配置")
        self.config_button.clicked.connect(self.show_camera_config)
        self.config_button.setMinimumHeight(40)
        controls_layout.addWidget(self.config_button)
        
        layout.addWidget(controls_group)
        
        # 配置信息
        config_group = QGroupBox("系统信息")
        config_layout = QVBoxLayout(config_group)
        
        self.config_label = QLabel()
        self.config_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        self.update_system_info()
        config_layout.addWidget(self.config_label)
        
        layout.addWidget(config_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 无限进度条
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
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
    
    def show_camera_config(self):
        """显示相机配置对话框"""
        dialog = CameraConfigDialog(self.camera_config, self)
        if dialog.exec_() == QDialog.Accepted:
            # 更新相机配置
            self.camera_config = dialog.get_config()
            self.log_text.append(f"相机配置已更新: 左目={self.camera_config['left_cam_id']}, 右目={self.camera_config['right_cam_id']}, 分辨率={self.camera_config['width']}x{self.camera_config['height']}")
            
            # 如果相机正在运行，提示重启
            if self.camera_worker is not None:
                self.log_text.append("警告: 需要重新启动相机匹配以应用新配置")
    
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
                free_mb = mem_info.free // (1024**2)
                
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
    
    def update_system_info(self):
        """更新系统信息显示"""
        gpu_info, gpu_memory_info = self.get_gpu_memory_info()
        
        config_text = f"""模型类型: {self.model_config['model_type']}
精度: {self.model_config['precision']}
置信度阈值: {self.model_config['conf_thresh']}
缩放因子: {self.model_config['resize_factor']}
GPU: {gpu_info}{gpu_memory_info}"""
        
        self.config_label.setText(config_text)
    
    def start_matching(self):
        """开始匹配"""
        if self.camera_worker is not None:
            return
            
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        
        # 创建工作线程
        self.camera_worker = CameraWorker(model_config=self.model_config)
        self.camera_worker.frame_ready.connect(self.update_display)
        self.camera_worker.status_update.connect(self.update_status)
        self.camera_worker.start()
        
        self.log_text.append("开始启动双目相机匹配...")
        
    def stop_matching(self):
        """停止匹配"""
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None
            
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        self.log_text.append("匹配已停止")
        
    def reset_stats(self):
        """重置统计信息"""
        self.frame_count = 0
        self.match_history.clear()
        self.fps_history.clear()
        
        self.matches_label.setText("匹配点数: 0")
        self.fps_label.setText("FPS: 0")
        self.avg_matches_label.setText("平均匹配: 0")
        self.avg_fps_label.setText("平均FPS: 0")
        
        self.log_text.append("统计信息已重置")
        
    def update_display(self, frame0, frame1, num_matches, fps, match_img=None):
        """更新显示"""
        self.frame_count += 1
        
        # 更新统计
        self.match_history.append(num_matches)
        self.fps_history.append(fps)
        
        if len(self.match_history) > 100:
            self.match_history.pop(0)
            self.fps_history.pop(0)
            
        avg_matches = np.mean(self.match_history)
        avg_fps = np.mean(self.fps_history)
        
        # 更新标签
        self.matches_label.setText(f"匹配点数: {num_matches}")
        self.fps_label.setText(f"FPS: {fps}")
        self.avg_matches_label.setText(f"平均匹配: {avg_matches:.1f}")
        self.avg_fps_label.setText(f"平均FPS: {avg_fps:.1f}")
        
        # 每10帧更新一次GPU信息（提高刷新频率）
        if self.frame_count % 10 == 0:
            self.update_system_info()
        
        # 显示相机图像
        self.display_frame(self.cam0_label, frame0)
        self.display_frame(self.cam1_label, frame1)
        
        # 显示匹配可视化
        if match_img is not None:
            self.display_frame(self.match_label, match_img)
        else:
            self.match_label.setText(f"匹配可视化\n匹配点数: {num_matches}")
        
    def display_frame(self, label, frame):
        """显示帧到标签"""
        try:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            # 缩放图像适应标签
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
        if self.camera_worker:
            self.camera_worker.stop()
        event.accept()

def main():
    parser = argparse.ArgumentParser(description='ELGS双目相机实时特征匹配')
    parser.add_argument('--cam0', type=int, default=0, help='第一个相机ID')
    parser.add_argument('--cam1', type=int, default=2, help='第二个相机ID')
    parser.add_argument('--model_type', choices=['full', 'opt'], default='full',
                       help='模型类型: full(最佳质量) 或 opt(最佳效率)')
    parser.add_argument('--precision', choices=['fp32', 'mp', 'fp16'], default='fp32',
                       help='计算精度')
    parser.add_argument('--conf_thresh', type=float, default=0.2,
                       help='置信度阈值')
    parser.add_argument('--resize_factor', type=float, default=0.8,
                       help='图像缩放因子')
    
    args = parser.parse_args()
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✓ 使用GPU: {torch.cuda.get_device_name()}")
    else:
        print("! 将使用CPU（性能较慢）")
    
    # 创建应用程序
    app = QApplication(sys.argv)
    
    # 创建主窗口
    window = DualCameraWindow()
    
    # 更新配置
    window.model_config.update({
        'cam0_id': args.cam0,
        'cam1_id': args.cam1,
        'model_type': args.model_type,
        'precision': args.precision,
        'conf_thresh': args.conf_thresh,
        'resize_factor': args.resize_factor
    })
    
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()