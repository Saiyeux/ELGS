#!/usr/bin/env python3
"""
配置对话框
允许用户编辑所有系统参数
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, 
                             QWidget, QLabel, QDoubleSpinBox, QSpinBox, 
                             QCheckBox, QComboBox, QPushButton, QGroupBox,
                             QGridLayout, QMessageBox, QLineEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from config.config_manager import ConfigManager

class ConfigDialog(QDialog):
    """配置参数对话框"""
    
    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setWindowTitle("ELGS - 系统参数配置")
        self.setModal(True)
        self.resize(600, 700)
        
        # 存储UI组件的引用
        self.ui_components = {}
        
        self.setup_ui()
        self.load_current_config()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # LoFTR配置标签页
        self.loftr_tab = self.create_loftr_tab()
        self.tab_widget.addTab(self.loftr_tab, "EfficientLoFTR")
        
        # Gaussian配置标签页
        self.gaussian_tab = self.create_gaussian_tab()
        self.tab_widget.addTab(self.gaussian_tab, "3D Gaussian Splatting")
        
        layout.addWidget(self.tab_widget)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("重置默认")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.apply_config)
        self.ok_btn.setDefault(True)
        
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.ok_btn)
        
        layout.addLayout(button_layout)
    
    def create_loftr_tab(self):
        """创建LoFTR配置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 基本参数组
        basic_group = QGroupBox("基本参数")
        basic_layout = QGridLayout(basic_group)
        
        row = 0
        
        # 启用/禁用
        basic_layout.addWidget(QLabel("启用LoFTR:"), row, 0)
        self.ui_components['loftr_enabled'] = QCheckBox()
        basic_layout.addWidget(self.ui_components['loftr_enabled'], row, 1)
        row += 1
        
        # 相机参数组
        camera_group = QGroupBox("相机参数")
        camera_layout = QGridLayout(camera_group)
        
        row = 0
        
        # 相机ID
        camera_layout.addWidget(QLabel("左相机ID:"), row, 0)
        self.ui_components['camera0_id'] = QSpinBox()
        self.ui_components['camera0_id'].setRange(0, 999)
        camera_layout.addWidget(self.ui_components['camera0_id'], row, 1)
        
        camera_layout.addWidget(QLabel("右相机ID:"), row, 2)
        self.ui_components['camera1_id'] = QSpinBox()
        self.ui_components['camera1_id'].setRange(0, 999)
        camera_layout.addWidget(self.ui_components['camera1_id'], row, 3)
        row += 1
        
        # 图像尺寸
        camera_layout.addWidget(QLabel("图像宽度:"), row, 0)
        self.ui_components['img_width'] = QSpinBox()
        self.ui_components['img_width'].setRange(1, 9999)
        self.ui_components['img_width'].setSingleStep(10)
        camera_layout.addWidget(self.ui_components['img_width'], row, 1)
        
        camera_layout.addWidget(QLabel("图像高度:"), row, 2)
        self.ui_components['img_height'] = QSpinBox()
        self.ui_components['img_height'].setRange(1, 9999)
        self.ui_components['img_height'].setSingleStep(10)
        camera_layout.addWidget(self.ui_components['img_height'], row, 3)
        row += 1
        
        # 焦距设置
        camera_layout.addWidget(QLabel("自动焦距:"), row, 0)
        self.ui_components['use_auto_focal_length'] = QCheckBox()
        self.ui_components['use_auto_focal_length'].toggled.connect(self.toggle_focal_length_mode)
        camera_layout.addWidget(self.ui_components['use_auto_focal_length'], row, 1)
        
        camera_layout.addWidget(QLabel("焦距比例:"), row, 2)
        self.ui_components['focal_length_scale'] = QDoubleSpinBox()
        self.ui_components['focal_length_scale'].setRange(0.001, 999.0)
        self.ui_components['focal_length_scale'].setSingleStep(0.1)
        self.ui_components['focal_length_scale'].setDecimals(2)
        camera_layout.addWidget(self.ui_components['focal_length_scale'], row, 3)
        row += 1
        
        # 手动焦距参数
        camera_layout.addWidget(QLabel("手动fx:"), row, 0)
        self.ui_components['manual_fx'] = QDoubleSpinBox()
        self.ui_components['manual_fx'].setRange(1, 99999)
        self.ui_components['manual_fx'].setDecimals(1)
        camera_layout.addWidget(self.ui_components['manual_fx'], row, 1)
        
        camera_layout.addWidget(QLabel("手动fy:"), row, 2)
        self.ui_components['manual_fy'] = QDoubleSpinBox()
        self.ui_components['manual_fy'].setRange(1, 99999)
        self.ui_components['manual_fy'].setDecimals(1)
        camera_layout.addWidget(self.ui_components['manual_fy'], row, 3)
        row += 1
        
        # 匹配参数组
        matching_group = QGroupBox("匹配参数")
        matching_layout = QGridLayout(matching_group)
        
        row = 0
        
        # 置信度阈值
        matching_layout.addWidget(QLabel("置信度阈值:"), row, 0)
        self.ui_components['conf_thresh'] = QDoubleSpinBox()
        self.ui_components['conf_thresh'].setRange(0.0, 10.0)
        self.ui_components['conf_thresh'].setSingleStep(0.05)
        self.ui_components['conf_thresh'].setDecimals(2)
        matching_layout.addWidget(self.ui_components['conf_thresh'], row, 1)
        
        # 缩放因子
        matching_layout.addWidget(QLabel("缩放因子:"), row, 2)
        self.ui_components['resize_factor'] = QDoubleSpinBox()
        self.ui_components['resize_factor'].setRange(0.1, 10.0)
        self.ui_components['resize_factor'].setSingleStep(0.05)
        self.ui_components['resize_factor'].setDecimals(2)
        matching_layout.addWidget(self.ui_components['resize_factor'], row, 3)
        row += 1
        
        # 跳帧设置
        matching_layout.addWidget(QLabel("跳帧数量:"), row, 0)
        self.ui_components['skip_frames'] = QSpinBox()
        self.ui_components['skip_frames'].setRange(0, 999)
        matching_layout.addWidget(self.ui_components['skip_frames'], row, 1)
        
        # 最小匹配点数
        matching_layout.addWidget(QLabel("最少匹配点:"), row, 2)
        self.ui_components['min_matches'] = QSpinBox()
        self.ui_components['min_matches'].setRange(1, 9999)
        matching_layout.addWidget(self.ui_components['min_matches'], row, 3)
        row += 1
        
        # 3D重建参数组
        reconstruction_group = QGroupBox("3D重建参数")
        reconstruction_layout = QGridLayout(reconstruction_group)
        
        row = 0
        
        # RANSAC参数
        reconstruction_layout.addWidget(QLabel("RANSAC置信度:"), row, 0)
        self.ui_components['ransac_confidence'] = QDoubleSpinBox()
        self.ui_components['ransac_confidence'].setRange(0.0, 1.0)
        self.ui_components['ransac_confidence'].setSingleStep(0.001)
        self.ui_components['ransac_confidence'].setDecimals(4)
        reconstruction_layout.addWidget(self.ui_components['ransac_confidence'], row, 1)
        
        reconstruction_layout.addWidget(QLabel("RANSAC阈值:"), row, 2)
        self.ui_components['ransac_threshold'] = QDoubleSpinBox()
        self.ui_components['ransac_threshold'].setRange(0.001, 999.0)
        self.ui_components['ransac_threshold'].setSingleStep(0.1)
        self.ui_components['ransac_threshold'].setDecimals(1)
        reconstruction_layout.addWidget(self.ui_components['ransac_threshold'], row, 3)
        row += 1
        
        # 深度过滤
        reconstruction_layout.addWidget(QLabel("深度阈值:"), row, 0)
        self.ui_components['depth_threshold'] = QDoubleSpinBox()
        self.ui_components['depth_threshold'].setRange(0.1, 9999.0)
        self.ui_components['depth_threshold'].setSingleStep(5.0)
        self.ui_components['depth_threshold'].setDecimals(1)
        reconstruction_layout.addWidget(self.ui_components['depth_threshold'], row, 1)
        row += 1
        
        # 组装布局
        layout.addWidget(basic_group)
        layout.addWidget(camera_group)
        layout.addWidget(matching_group)
        layout.addWidget(reconstruction_group)
        layout.addStretch()
        
        return tab
    
    def create_gaussian_tab(self):
        """创建Gaussian配置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 基本参数组
        basic_group = QGroupBox("基本参数")
        basic_layout = QGridLayout(basic_group)
        
        row = 0
        
        # 启用/禁用
        basic_layout.addWidget(QLabel("启用实时重建:"), row, 0)
        self.ui_components['gaussian_enabled'] = QCheckBox()
        basic_layout.addWidget(self.ui_components['gaussian_enabled'], row, 1)
        
        # 实时重建
        basic_layout.addWidget(QLabel("实时处理:"), row, 2)
        self.ui_components['realtime_enabled'] = QCheckBox()
        basic_layout.addWidget(self.ui_components['realtime_enabled'], row, 3)
        row += 1
        
        # 像素缩放
        basic_layout.addWidget(QLabel("像素缩放:"), row, 0)
        self.ui_components['pixel_scale'] = QDoubleSpinBox()
        self.ui_components['pixel_scale'].setRange(0.000001, 1.0)
        self.ui_components['pixel_scale'].setSingleStep(0.0001)
        self.ui_components['pixel_scale'].setDecimals(4)
        basic_layout.addWidget(self.ui_components['pixel_scale'], row, 1)
        
        # 最少点数
        basic_layout.addWidget(QLabel("最少点数:"), row, 2)
        self.ui_components['min_points'] = QSpinBox()
        self.ui_components['min_points'].setRange(1, 99999)
        self.ui_components['min_points'].setSingleStep(1)
        basic_layout.addWidget(self.ui_components['min_points'], row, 3)
        row += 1
        
        # 重建参数组
        training_group = QGroupBox("重建参数")
        training_layout = QGridLayout(training_group)
        
        row = 0
        
        # 迭代次数（1-1000）
        training_layout.addWidget(QLabel("迭代次数:"), row, 0)
        self.ui_components['iterations'] = QSpinBox()
        self.ui_components['iterations'].setRange(1, 999999)  # 取消上限限制
        self.ui_components['iterations'].setSingleStep(10)
        training_layout.addWidget(self.ui_components['iterations'], row, 1)
        
        # 虚拟相机数量
        training_layout.addWidget(QLabel("虚拟相机数:"), row, 2)
        self.ui_components['num_cameras'] = QSpinBox()
        self.ui_components['num_cameras'].setRange(1, 99)  # 取消限制
        training_layout.addWidget(self.ui_components['num_cameras'], row, 3)
        row += 1
        
        # 学习率参数
        training_layout.addWidget(QLabel("位置学习率:"), row, 0)
        self.ui_components['position_lr_init'] = QDoubleSpinBox()
        self.ui_components['position_lr_init'].setRange(0.000001, 1.0)
        self.ui_components['position_lr_init'].setSingleStep(0.00001)
        self.ui_components['position_lr_init'].setDecimals(5)
        training_layout.addWidget(self.ui_components['position_lr_init'], row, 1)
        
        training_layout.addWidget(QLabel("特征学习率:"), row, 2)
        self.ui_components['feature_lr'] = QDoubleSpinBox()
        self.ui_components['feature_lr'].setRange(0.000001, 1.0)
        self.ui_components['feature_lr'].setSingleStep(0.0005)
        self.ui_components['feature_lr'].setDecimals(4)
        training_layout.addWidget(self.ui_components['feature_lr'], row, 3)
        row += 1
        
        # 密度增强参数组
        density_group = QGroupBox("密度增强参数")
        density_layout = QGridLayout(density_group)
        
        row = 0
        
        # 密度增强开关
        density_layout.addWidget(QLabel("启用密度增强:"), row, 0)
        self.ui_components['density_enhancement'] = QCheckBox()
        density_layout.addWidget(self.ui_components['density_enhancement'], row, 1)
        
        # 增强倍数
        density_layout.addWidget(QLabel("增强倍数:"), row, 2)
        self.ui_components['enhancement_factor'] = QSpinBox()
        self.ui_components['enhancement_factor'].setRange(1, 9999)
        self.ui_components['enhancement_factor'].setSingleStep(1)
        density_layout.addWidget(self.ui_components['enhancement_factor'], row, 3)
        row += 1
        
        # 梯度阈值
        density_layout.addWidget(QLabel("密度梯度阈值:"), row, 0)
        self.ui_components['densify_grad_threshold'] = QDoubleSpinBox()
        self.ui_components['densify_grad_threshold'].setRange(0.000001, 1.0)
        self.ui_components['densify_grad_threshold'].setSingleStep(0.0001)
        self.ui_components['densify_grad_threshold'].setDecimals(4)
        density_layout.addWidget(self.ui_components['densify_grad_threshold'], row, 1)
        
        # Lambda DSSIM
        density_layout.addWidget(QLabel("DSSIM权重:"), row, 2)
        self.ui_components['lambda_dssim'] = QDoubleSpinBox()
        self.ui_components['lambda_dssim'].setRange(0.0, 10.0)
        self.ui_components['lambda_dssim'].setSingleStep(0.05)
        self.ui_components['lambda_dssim'].setDecimals(2)
        density_layout.addWidget(self.ui_components['lambda_dssim'], row, 3)
        row += 1
        
        # 组装布局
        layout.addWidget(basic_group)
        layout.addWidget(training_group)
        layout.addWidget(density_group)
        layout.addStretch()
        
        return tab
    
    def toggle_focal_length_mode(self, auto_mode):
        """切换焦距模式"""
        self.ui_components['focal_length_scale'].setEnabled(auto_mode)
        self.ui_components['manual_fx'].setEnabled(not auto_mode)
        self.ui_components['manual_fy'].setEnabled(not auto_mode)
    
    def load_current_config(self):
        """加载当前配置到UI"""
        ui_config = self.config_manager.get_ui_config_dict()
        
        # 加载LoFTR配置
        loftr_config = ui_config.get('loftr', {})
        for key, component in self.ui_components.items():
            if key.startswith('loftr_') or key in loftr_config:
                config_key = key.replace('loftr_', '') if key.startswith('loftr_') else key
                if config_key in loftr_config:
                    value = loftr_config[config_key]
                    
                    if isinstance(component, QCheckBox):
                        component.setChecked(value)
                    elif isinstance(component, (QSpinBox, QDoubleSpinBox)):
                        component.setValue(value)
                    elif isinstance(component, QComboBox):
                        index = component.findText(str(value))
                        if index >= 0:
                            component.setCurrentIndex(index)
        
        # 加载Gaussian配置
        gaussian_config = ui_config.get('gaussian', {})
        for key, component in self.ui_components.items():
            if key.startswith('gaussian_') or key in gaussian_config:
                config_key = key.replace('gaussian_', '') if key.startswith('gaussian_') else key
                if config_key in gaussian_config:
                    value = gaussian_config[config_key]
                    
                    if isinstance(component, QCheckBox):
                        component.setChecked(value)
                    elif isinstance(component, (QSpinBox, QDoubleSpinBox)):
                        component.setValue(value)
                    elif isinstance(component, QComboBox):
                        index = component.findText(str(value))
                        if index >= 0:
                            component.setCurrentIndex(index)
        
        # 更新焦距模式显示
        auto_focal = self.ui_components.get('use_auto_focal_length')
        if auto_focal:
            self.toggle_focal_length_mode(auto_focal.isChecked())
    
    def collect_ui_config(self):
        """收集UI中的配置"""
        loftr_config = {}
        gaussian_config = {}
        
        for key, component in self.ui_components.items():
            if isinstance(component, QCheckBox):
                value = component.isChecked()
            elif isinstance(component, (QSpinBox, QDoubleSpinBox)):
                value = component.value()
            elif isinstance(component, QComboBox):
                value = component.currentText()
            else:
                continue
            
            if key.startswith('loftr_'):
                config_key = key.replace('loftr_', '')
                loftr_config[config_key] = value
            elif key.startswith('gaussian_'):
                config_key = key.replace('gaussian_', '')
                gaussian_config[config_key] = value
            else:
                # 根据当前标签页决定配置归属
                current_tab = self.tab_widget.currentIndex()
                if current_tab == 0:  # LoFTR标签页
                    loftr_config[key] = value
                else:  # Gaussian标签页
                    gaussian_config[key] = value
        
        return {'loftr': loftr_config, 'gaussian': gaussian_config}
    
    def apply_config(self):
        """应用配置"""
        try:
            # 收集UI配置
            ui_config_dict = self.collect_ui_config()
            
            # 应用到配置管理器
            self.config_manager.apply_ui_config_dict(ui_config_dict)
            
            # 验证配置
            if not self.config_manager.validate_all_configs():
                QMessageBox.warning(self, "配置验证", "配置参数验证失败，请检查参数范围")
                return
            
            # 保存配置
            if self.config_manager.save_all_configs():
                QMessageBox.information(self, "配置保存", "配置已成功保存")
                self.accept()
            else:
                QMessageBox.warning(self, "保存失败", "配置保存失败，请检查文件权限")
                
        except Exception as e:
            QMessageBox.critical(self, "应用配置失败", f"应用配置时发生错误：{str(e)}")
    
    def reset_to_defaults(self):
        """重置为默认配置"""
        reply = QMessageBox.question(self, "重置配置", 
                                   "确定要将所有配置重置为默认值吗？此操作不可撤销。",
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.config_manager.reset_all_to_defaults()
            self.load_current_config()
            QMessageBox.information(self, "重置完成", "配置已重置为默认值")