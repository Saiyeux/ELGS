#!/usr/bin/env python3
import sys
import time
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                             QTextEdit, QGroupBox, QCheckBox, QSpinBox)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from pathlib import Path

from EDGS.model.camera_thread import CameraThread
from EDGS.model.matching_thread import MatchingThread
from EDGS.model.gaussian_thread import GaussianThread
from EDGS.model.filter_thread import FilterThread
from EDGS.control.video_widget import VideoWidget
from EDGS.control.point3d_widget import Point3DWidget
from EDGS.control.config_dialog import ConfigDialog
from config.config_manager import ConfigManager

class ELGSMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Efficient LoFTR with 3D Gaussian Splatting")
        self.setGeometry(100, 100, 1600, 1200)
        
        # 配置管理器
        self.config_manager = ConfigManager()
        
        # Threads
        self.camera0_thread = CameraThread(0)
        self.camera1_thread = CameraThread(2)
        self.matching_thread = MatchingThread()
        self.gaussian_thread = GaussianThread()
        self.filter_thread = FilterThread()
        
        # Data storage
        self.frame0 = None
        self.frame1 = None
        self.current_matches = None
        
        self.setup_ui()
        self.connect_signals()
        self.load_initial_config()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QGridLayout(central_widget)
        
        # Top: Camera views
        camera_group = QGroupBox("双目相机视图")
        camera_layout = QHBoxLayout(camera_group)
        
        self.video_widget0 = VideoWidget("左相机")
        self.video_widget1 = VideoWidget("右相机")
        
        camera_layout.addWidget(self.video_widget0)
        camera_layout.addWidget(self.video_widget1)
        
        # Bottom left: 3D reconstruction
        self.point3d_widget = Point3DWidget()
        
        # Bottom right: Simple controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Simple control buttons
        controls_group = QGroupBox("系统控制")
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setSpacing(8)  # Add spacing between controls
        
        # Main control buttons
        self.start_btn = QPushButton("启动")
        self.start_btn.setMinimumHeight(45)
        self.start_btn.setStyleSheet("QPushButton { font-size: 16px; background-color: #4CAF50; color: white; border-radius: 8px; font-weight: bold; }")
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setMinimumHeight(45)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { font-size: 16px; background-color: #f44336; color: white; border-radius: 8px; font-weight: bold; }")
        
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        
        # Add separator
        controls_layout.addSpacing(10)
        
        # 配置按钮
        self.config_btn = QPushButton("参数配置")
        self.config_btn.setMinimumHeight(35)
        self.config_btn.setStyleSheet("QPushButton { font-size: 13px; background-color: #2196F3; color: white; border-radius: 6px; }")
        controls_layout.addWidget(self.config_btn)
        
        # 重置世界坐标系按钮
        self.reset_world_btn = QPushButton("重置坐标系")
        self.reset_world_btn.setMinimumHeight(35)
        self.reset_world_btn.setStyleSheet("QPushButton { font-size: 13px; background-color: #FF9800; color: white; border-radius: 6px; }")
        self.reset_world_btn.setToolTip("重置世界坐标系原点，消除点云漂移")
        controls_layout.addWidget(self.reset_world_btn)
        
        controls_layout.addStretch()
        
        right_layout.addWidget(controls_group)
        
        # Advanced processing controls
        advanced_group = QGroupBox("高级处理")
        advanced_layout = QVBoxLayout(advanced_group)
        
        # Noise filtering controls
        filter_group = QGroupBox("匹配滤波")
        filter_layout = QVBoxLayout(filter_group)
        
        self.filter_enable_cb = QCheckBox("启用噪声滤波")
        self.filter_enable_cb.setChecked(True)
        filter_layout.addWidget(self.filter_enable_cb)
        
        # Filter status label
        self.filter_status_label = QLabel("状态: 就绪")
        self.filter_status_label.setStyleSheet("QLabel { font-size: 10px; color: #666; }")
        filter_layout.addWidget(self.filter_status_label)
        
        advanced_layout.addWidget(filter_group)
        
        # Gaussian splatting controls
        gaussian_group = QGroupBox("高斯实时重建")
        gaussian_layout = QVBoxLayout(gaussian_group)
        
        self.gaussian_enable_cb = QCheckBox("启用")
        self.gaussian_enable_cb.setChecked(False)
        gaussian_layout.addWidget(self.gaussian_enable_cb)
        
        # Reconstruction parameters
        recon_params_layout = QHBoxLayout()
        
        # Reconstruction iterations
        recon_params_layout.addWidget(QLabel("迭代:"))
        self.training_iterations_spin = QSpinBox()
        self.training_iterations_spin.setRange(1, 999999)  # 取消上限限制，允许任意大的迭代次数
        self.training_iterations_spin.setValue(100)      # 默认200
        self.training_iterations_spin.setSuffix(" 次")
        recon_params_layout.addWidget(self.training_iterations_spin)
        
        gaussian_layout.addLayout(recon_params_layout)
        
        # Manual reconstruction button
        self.manual_recon_btn = QPushButton("立即重建")
        self.manual_recon_btn.setEnabled(False)
        self.manual_recon_btn.setStyleSheet("QPushButton { font-size: 12px; }")
        gaussian_layout.addWidget(self.manual_recon_btn)
        
        # Gaussian status label
        self.gaussian_status_label = QLabel("状态: 就绪")
        self.gaussian_status_label.setStyleSheet("QLabel { font-size: 10px; color: #666; }")
        gaussian_layout.addWidget(self.gaussian_status_label)
        
        advanced_layout.addWidget(gaussian_group)
        
        right_layout.addWidget(advanced_group)
        
        # Error log output (minimal)
        log_group = QGroupBox("log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(350)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("QTextEdit { background-color: #1a1a1a; color: #ff6b6b; }")
        log_layout.addWidget(self.log_text)
        
        right_layout.addWidget(log_group)
        
        # Layout arrangement
        main_layout.addWidget(camera_group, 0, 0, 1, 2)
        main_layout.addWidget(self.point3d_widget, 1, 0)
        main_layout.addWidget(right_widget, 1, 1)
        
        # Set row/column stretch - expand stereo display area
        main_layout.setRowStretch(0, 3)
        main_layout.setRowStretch(1, 1)
        main_layout.setColumnStretch(0, 2)
        main_layout.setColumnStretch(1, 1)
        
    def connect_signals(self):
        # Button connections
        self.start_btn.clicked.connect(self.start_system)
        self.stop_btn.clicked.connect(self.stop_system)
        self.config_btn.clicked.connect(self.open_config_dialog)
        self.reset_world_btn.clicked.connect(self.reset_world_coordinate_system)
        
        # Camera threads
        self.camera0_thread.frame_ready.connect(self.update_camera_frame)
        self.camera1_thread.frame_ready.connect(self.update_camera_frame)
        
        # Matching thread
        self.matching_thread.matches_ready.connect(self.update_matches)
        self.matching_thread.points_3d_ready.connect(self.update_3d_points)
        self.matching_thread.log_message.connect(self.add_error_log)
        
        # Advanced processing controls
        self.filter_enable_cb.toggled.connect(self.toggle_filter)
        self.gaussian_enable_cb.toggled.connect(self.toggle_gaussian)
        self.training_iterations_spin.valueChanged.connect(self.update_reconstruction_params)
        self.manual_recon_btn.clicked.connect(self.start_manual_reconstruction)
        
        # Filter thread signals
        self.filter_thread.matches_filtered.connect(self.handle_filtered_matches)
        self.filter_thread.filter_status_updated.connect(self.update_filter_status)
        self.filter_thread.error_occurred.connect(self.add_error_log)
        
        # Gaussian thread signals
        self.gaussian_thread.reconstruction_started.connect(self.on_gaussian_reconstruction_started)
        self.gaussian_thread.reconstruction_completed.connect(self.on_gaussian_reconstruction_completed)
        self.gaussian_thread.reconstruction_failed.connect(self.on_gaussian_reconstruction_failed)
        self.gaussian_thread.status_updated.connect(self.update_gaussian_status)
        self.gaussian_thread.dense_points_ready.connect(self.update_dense_points)
        
    def start_system(self):
        # Use fixed camera IDs (based on system test results)
        self.camera0_thread.camera_id = 0
        self.camera1_thread.camera_id = 4
        
        # Start cameras
        if not self.camera0_thread.start_camera():
            self.add_error_log(f"错误：无法打开左相机 (ID: 0)")
            return
            
        if not self.camera1_thread.start_camera():
            self.add_error_log(f"错误：无法打开右相机 (ID: 2)")
            self.camera0_thread.stop_camera()
            return
            
        # Start matching
        self.matching_thread.start_matching()
        
        # Initialize advanced processing threads with current settings
        self.filter_thread.set_enabled(self.filter_enable_cb.isChecked())
        self.gaussian_thread.set_enabled(self.gaussian_enable_cb.isChecked())
        self.gaussian_thread.set_reconstruction_parameters(iterations=self.training_iterations_spin.value())
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
    def stop_system(self):
        self.camera0_thread.stop_camera()
        self.camera1_thread.stop_camera()
        self.matching_thread.stop_matching()
        
        # Stop advanced processing threads
        self.filter_thread.set_enabled(False)
        self.filter_thread.clear_queue()
        self.gaussian_thread.set_enabled(False)
        self.gaussian_thread.clear_queue()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def reset_world_coordinate_system(self):
        """重置世界坐标系，消除点云漂移"""
        try:
            # 重置匹配线程的世界坐标系
            self.matching_thread.reset_world_coordinate_system()
            
            # 清除当前显示的点云和颜色信息
            self.point3d_widget.points_3d = None
            self.point3d_widget.dense_points_3d = None
            self.point3d_widget.points_colors = None
            self.point3d_widget.dense_points_colors = None
            self.point3d_widget.redraw_all_points()
            
            self.add_error_log("世界坐标系已重置，请等待重新校准...")
            
        except Exception as e:
            self.add_error_log(f"重置坐标系失败: {str(e)}")
        
    def update_camera_frame(self, frame, camera_id):
        if camera_id == self.camera0_thread.camera_id:
            self.frame0 = frame
            self.video_widget0.update_frame(frame)
        elif camera_id == self.camera1_thread.camera_id:
            self.frame1 = frame
            self.video_widget1.update_frame(frame)
            
        # Update matching thread with both frames
        if self.frame0 is not None and self.frame1 is not None:
            self.matching_thread.update_frames(self.frame0, self.frame1)
            
    def update_matches(self, mkpts0, mkpts1, mconf):
        self.current_matches = (mkpts0, mkpts1, mconf)
        
        # Send matches to filter thread for processing
        if self.filter_enable_cb.isChecked():
            self.filter_thread.add_matches(mkpts0, mkpts1, mconf)
        
        # Update video widgets with matches
        self.video_widget0.update_matches(mkpts0, 0)
        self.video_widget1.update_matches(mkpts1, 1)
        
    def update_3d_points(self, points_3d, colors_rgb):
        self.point3d_widget.update_points(points_3d, colors_rgb)
        
        # Send 3D points to Gaussian thread for dense reconstruction
        if self.gaussian_enable_cb.isChecked() and colors_rgb is not None:
            self.gaussian_thread.add_points(points_3d, colors_rgb)
            
    def add_error_log(self, message):
        """只记录错误和警告信息"""
        if "错误" in message or "失败" in message or "警告" in message or "Error" in message:
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            self.log_text.append(log_entry)
            self.log_text.ensureCursorVisible()
        
    def closeEvent(self, event):
        self.stop_system()
        event.accept()
    
    # Advanced processing methods
    def toggle_filter(self, enabled):
        """切换匹配滤波"""
        self.filter_thread.set_enabled(enabled)
        if enabled:
            self.filter_status_label.setText("状态: 已启用")
            self.filter_status_label.setStyleSheet("QLabel { font-size: 10px; color: #4CAF50; }")
        else:
            self.filter_status_label.setText("状态: 已禁用")
            self.filter_status_label.setStyleSheet("QLabel { font-size: 10px; color: #666; }")
    
    def toggle_gaussian(self, enabled):
        """切换高斯实时重建"""
        self.gaussian_thread.set_enabled(enabled)
        self.manual_recon_btn.setEnabled(enabled)
        self.training_iterations_spin.setEnabled(enabled)
        
        if enabled:
            self.gaussian_status_label.setText("状态: 已启用")
            self.gaussian_status_label.setStyleSheet("QLabel { font-size: 10px; color: #4CAF50; }")
        else:
            self.gaussian_status_label.setText("状态: 已禁用")
            self.gaussian_status_label.setStyleSheet("QLabel { font-size: 10px; color: #666; }")
    
    def update_reconstruction_params(self):
        """更新重建参数"""
        iterations = self.training_iterations_spin.value()
        self.gaussian_thread.set_reconstruction_parameters(iterations=iterations)
    
    def start_manual_reconstruction(self):
        """手动开始重建"""
        if self.gaussian_thread.is_available():
            iterations = self.training_iterations_spin.value()
            success = self.gaussian_thread.start_reconstruction(iterations=iterations)
            if not success:
                self.add_error_log("错误：无法启动高斯重建")
        else:
            self.add_error_log("错误：高斯重建功能不可用")
    
    def handle_filtered_matches(self, pts0, pts1, stats):
        """处理过滤后的匹配点"""
        # 更新当前匹配数据
        if len(pts0) > 0:
            # 使用过滤后的匹配点进行后续处理
            # 这里可以替换原始匹配数据或作为额外输入
            pass
    
    def update_filter_status(self, status):
        """更新滤波器状态"""
        if status.get('enabled', False):
            processed = status.get('stats', {}).get('total_processed', 0)
            avg_ratio = status.get('stats', {}).get('average_filter_ratio', 0.0)
            processing_time = status.get('stats', {}).get('processing_time_ms', 0.0)
            
            status_text = f"已处理: {processed} 帧 | 保留率: {avg_ratio:.2f} | 耗时: {processing_time:.1f}ms"
            self.filter_status_label.setText(status_text)
    
    def on_gaussian_reconstruction_started(self):
        """高斯重建开始"""
        self.gaussian_status_label.setText("状态: 重建中...")
        self.gaussian_status_label.setStyleSheet("QLabel { font-size: 10px; color: #FF9800; }")
        self.manual_recon_btn.setEnabled(False)
    
    def on_gaussian_reconstruction_completed(self, dense_points):
        """高斯重建完成"""
        point_count = len(dense_points) if dense_points is not None else 0
        self.gaussian_status_label.setText(f"状态: 重建完成 ({point_count} 点)")
        self.gaussian_status_label.setStyleSheet("QLabel { font-size: 10px; color: #4CAF50; }")
        self.manual_recon_btn.setEnabled(True)
    
    def on_gaussian_reconstruction_failed(self, error_msg):
        """高斯重建失败"""
        self.gaussian_status_label.setText("状态: 重建失败")
        self.gaussian_status_label.setStyleSheet("QLabel { font-size: 10px; color: #f44336; }")
        self.manual_recon_btn.setEnabled(True)
        self.add_error_log(f"错误：高斯重建失败 - {error_msg}")
    
    def update_dense_points(self, dense_points, dense_colors):
        """更新稠密点云显示"""
        if dense_points is not None and len(dense_points) > 0:
            # 在3D视图中同时显示稀疏和稠密点云
            # 这里我们需要修改Point3DWidget以支持多层点云显示
            self.point3d_widget.update_dense_points(dense_points, dense_colors)
    
    def update_gaussian_status(self, status):
        """更新高斯重建状态"""
        if status.get('enabled', False):
            is_reconstructing = status.get('is_reconstructing', False)
            has_dense_points = status.get('has_dense_points', False)
            dense_count = status.get('dense_point_count', 0)
            
            if is_reconstructing:
                self.gaussian_status_label.setText("状态: 重建中...")
                self.gaussian_status_label.setStyleSheet("QLabel { font-size: 10px; color: #FF9800; }")
            elif has_dense_points:
                status_text = f"状态: 稠密点云 {dense_count} 点"
                self.gaussian_status_label.setText(status_text)
                self.gaussian_status_label.setStyleSheet("QLabel { font-size: 10px; color: #4CAF50; }")
            else:
                self.gaussian_status_label.setText("状态: 等待点云数据")
                self.gaussian_status_label.setStyleSheet("QLabel { font-size: 10px; color: #666; }")
    
    # Configuration methods
    def load_initial_config(self):
        """加载初始配置到UI"""
        try:
            ui_config = self.config_manager.get_ui_config_dict()
            
            # 加载LoFTR配置
            loftr_config = ui_config.get('loftr', {})
            
            # 加载Gaussian配置  
            gaussian_config = ui_config.get('gaussian', {})
            
            # 应用到UI控件
            self.filter_enable_cb.setChecked(loftr_config.get('enable_stable_filter', True))
            self.gaussian_enable_cb.setChecked(gaussian_config.get('enabled', False))
            self.training_iterations_spin.setValue(gaussian_config.get('iterations', 200))
            
            # 更新线程配置
            self.update_thread_configs()
            
        except Exception as e:
            self.add_error_log(f"配置加载失败: {e}")
    
    def update_thread_configs(self):
        """根据当前配置更新线程参数"""
        try:
            # 更新LoFTR线程配置
            loftr_config = self.config_manager.loftr_config
            
            # 更新相机ID
            self.camera0_thread.camera_id = loftr_config.get('camera.camera0_id', 0)
            self.camera1_thread.camera_id = loftr_config.get('camera.camera1_id', 2)
            
            # 更新匹配线程参数
            self.matching_thread.conf_thresh = loftr_config.get('matching.conf_thresh', 0.2)
            self.matching_thread.resize_factor = loftr_config.get('matching.resize_factor', 0.8)
            
            # 更新相机矩阵
            camera_matrix = loftr_config.get_camera_matrix()
            self.matching_thread.camera_matrix = camera_matrix
            
            # 更新图像尺寸
            self.matching_thread.img_width = loftr_config.get('camera.img_width', 640)
            self.matching_thread.img_height = loftr_config.get('camera.img_height', 480)
            
            # 更新Gaussian线程配置
            gaussian_config = self.config_manager.gaussian_config
            
            self.gaussian_thread.pixel_scale = gaussian_config.get('pixel_scale', 0.001)
            
            # 更新重建参数
            iterations = gaussian_config.get('reconstruction.iterations', 200)
            self.gaussian_thread.set_reconstruction_parameters(iterations=iterations)
            
        except Exception as e:
            self.add_error_log(f"线程配置更新失败: {e}")
    
    def open_config_dialog(self):
        """打开配置对话框"""
        try:
            dialog = ConfigDialog(self.config_manager, self)
            if dialog.exec_() == dialog.Accepted:
                # 配置已在对话框中保存，重新加载配置
                self.config_manager.load_all_configs()
                self.load_initial_config()
                self.add_error_log("信息: 配置已更新并应用")
        except Exception as e:
            self.add_error_log(f"打开配置对话框失败: {e}")