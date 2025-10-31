#!/usr/bin/env python3
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Point3DWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b;")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title with point count
        self.title_label = QLabel("3D点云视图")
        self.title_label.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        # 3D visualization using matplotlib
        from matplotlib.backends.qt_compat import QtWidgets
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from mpl_toolkits.mplot3d import Axes3D
        
        self.figure = Figure(figsize=(6, 6), facecolor='#2b2b2b')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #2b2b2b;")
        
        # 禁用matplotlib的导航工具栏，防止用户意外改变视图
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        
        # Create 3D subplot
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#1a1a1a')
        
        # Enable mouse interaction for rotation
        self.ax.mouse_init()
        
        # Set fixed scale and axes
        self.setup_3d_axes()
        
        # 初始化时强制设置一致的坐标系
        self.force_consistent_redraw()
        
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.points_3d = None
        self.dense_points_3d = None
        self.points_colors = None  # 稀疏点云颜色
        self.dense_points_colors = None  # 稠密点云颜色
        self.scatter = None
        self.dense_scatter = None
        self.colorbar = None
        
        # 防止频繁重绘的标志
        self._drawing = False
        
        # 统一的坐标变换参数
        self.coordinate_transform_params = {
            'x_flip': True,
            'y_flip': True, 
            'z_flip': True,
            'z_offset': 2.5
        }
        
    def transform_coordinates(self, points_3d):
        """统一的坐标变换函数，确保稀疏和稠密点云使用相同的坐标系"""
        if points_3d is None or len(points_3d) == 0:
            return points_3d, None
            
        points_transformed = points_3d.copy()
        
        # 应用统一的坐标变换
        if self.coordinate_transform_params['x_flip']:
            points_transformed[:, 0] = -points_3d[:, 0]
        if self.coordinate_transform_params['y_flip']:
            points_transformed[:, 1] = -points_3d[:, 1]
        if self.coordinate_transform_params['z_flip']:
            points_transformed[:, 2] = -points_3d[:, 2] + self.coordinate_transform_params['z_offset']
        
        # 严格过滤超出坐标系范围的点，留出极小边距确保不触碰边界
        # 确保超出坐标系的点完全舍弃，避免任何边界堆积
        valid_mask = (
            (points_transformed[:, 0] > -0.48) & (points_transformed[:, 0] < 0.48) &
            (points_transformed[:, 1] > -0.48) & (points_transformed[:, 1] < 0.48) &
            (points_transformed[:, 2] > 0.52) & (points_transformed[:, 2] < 1.98)
        )
        
        if np.any(valid_mask):
            return points_transformed[valid_mask], valid_mask
        else:
            return np.array([]).reshape(0, 3), None
        
    def _ensure_fixed_axes_settings(self):
        """确保坐标轴设置固定，不触发重绘"""
        # 设置固定的坐标轴范围和属性
        self.ax.set_xlim([-0.5, 0.5])
        self.ax.set_ylim([-0.5, 0.5]) 
        self.ax.set_zlim([0.5, 2.0])
        self.ax.set_box_aspect([1, 1, 1.5])
        self.ax.view_init(elev=25, azim=45)
        
        # 禁用所有自动缩放
        self.ax.autoscale(False)
        self.ax.set_autoscale_on(False)
        self.ax.set_autoscalex_on(False) 
        self.ax.set_autoscaley_on(False)
        self.ax.set_autoscalez_on(False)
        
    def force_consistent_redraw(self):
        """强制进行一致的重绘，确保坐标系和缩放始终相同"""
        # 确保坐标轴设置
        self._ensure_fixed_axes_settings()
        
        # 执行重绘
        self.canvas.draw()
        
    def setup_3d_axes(self):
        """设置固定比例的3D坐标轴 - 确保始终保持一致"""
        # Set axis labels
        self.ax.set_xlabel('X', color='white', fontsize=10)
        self.ax.set_ylabel('Y', color='white', fontsize=10)
        self.ax.set_zlabel('Z', color='white', fontsize=10)
        
        # Set FIXED axis limits - 这些限制永远不变，无论是否有稠密点云
        self.ax.set_xlim([-0.5, 0.5])
        self.ax.set_ylim([-0.5, 0.5])
        self.ax.set_zlim([0.5, 2.0])
        
        # Force equal aspect ratio - 固定比例防止变形
        self.ax.set_box_aspect([1, 1, 1.5])
        
        # Customize axis appearance
        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        
        # Set grid
        self.ax.grid(True, alpha=0.3)
        
        # 应用统一的坐标轴设置
        self._ensure_fixed_axes_settings()
        
        self.figure.tight_layout()
        # 注意：不在这里调用canvas.draw()以避免重复绘制
        
    def update_points(self, points_3d, colors_rgb=None):
        """更新3D稀疏点云显示 - 避免闪烁的智能更新"""
        self.points_3d = points_3d
        self.points_colors = colors_rgb  # 保存真实像素颜色
        
        # 使用统一的重绘方法，避免分别更新造成闪烁
        self.redraw_all_points()
    
    def redraw_all_points(self):
        """重新绘制所有点云（稀疏和稠密） - 防闪烁优化版本"""
        # 防止重复重绘
        if self._drawing:
            return
        self._drawing = True
        
        try:
            # 清除colorbar（不再使用depth colorbar）
            if self.colorbar is not None:
                self.colorbar.remove()
                self.colorbar = None
            
            # 清除现有的散点图对象，但避免清除整个axes
            if self.scatter is not None:
                self.scatter.remove()
                self.scatter = None
            if self.dense_scatter is not None:
                self.dense_scatter.remove() 
                self.dense_scatter = None
                
            # 清除legend（如果存在）
            if self.ax.legend_ is not None:
                self.ax.legend_.remove()
            
            # 重新确保坐标轴设置（不清除axes）
            self._ensure_fixed_axes_settings()
            
            # 绘制稀疏点云
            if self.points_3d is not None and len(self.points_3d) > 0:
                # 使用统一的坐标变换函数
                points_transformed, valid_mask = self.transform_coordinates(self.points_3d)
                
                if len(points_transformed) > 0:
                    # 使用真实像素颜色而不是深度着色
                    if hasattr(self, 'points_colors') and self.points_colors is not None and valid_mask is not None:
                        # 对应地过滤颜色数据
                        colors = self.points_colors[valid_mask] / 255.0
                        self.scatter = self.ax.scatter(points_transformed[:, 0], points_transformed[:, 1], points_transformed[:, 2], 
                                                     c=colors, s=15, alpha=0.8)
                    else:
                        # 如果没有颜色信息，使用默认颜色
                        self.scatter = self.ax.scatter(points_transformed[:, 0], points_transformed[:, 1], points_transformed[:, 2], 
                                                     c='red', s=15, alpha=0.8)
            
            # 绘制稠密点云
            if self.dense_points_3d is not None and len(self.dense_points_3d) > 0:
                # 使用统一的坐标变换函数
                dense_transformed, dense_valid_mask = self.transform_coordinates(self.dense_points_3d)
                
                if len(dense_transformed) > 0:
                    # 使用真实像素颜色
                    if hasattr(self, 'dense_points_colors') and self.dense_points_colors is not None and dense_valid_mask is not None:
                        dense_colors = self.dense_points_colors[dense_valid_mask] / 255.0
                        self.dense_scatter = self.ax.scatter(dense_transformed[:, 0], dense_transformed[:, 1], dense_transformed[:, 2], 
                                                           c=dense_colors, s=3, alpha=0.6)
                    else:
                        # 如果没有颜色信息，使用青色
                        self.dense_scatter = self.ax.scatter(dense_transformed[:, 0], dense_transformed[:, 1], dense_transformed[:, 2], 
                                                           c='cyan', s=3, alpha=0.4)
            
            # 绘制完成后，立即确保坐标轴设置固定（防止scatter导致的自动调整）
            self._ensure_fixed_axes_settings()
            
            # 更新标题（显示总点数，不区分稠密稀疏）
            sparse_count = len(self.points_3d) if self.points_3d is not None else 0
            dense_count = len(self.dense_points_3d) if self.dense_points_3d is not None else 0
            total_count = sparse_count + dense_count
            
            self.title_label.setText(f"3D点云视图(点数:{total_count})")
            
            # 最后再次确保坐标系固定，然后重绘
            self.force_consistent_redraw()
            
        finally:
            # 重置绘制标志
            self._drawing = False
    
    def update_dense_points(self, dense_points_3d, dense_colors_rgb=None):
        """更新稠密点云显示（与稀疏点云同时显示）"""
        self.dense_points_3d = dense_points_3d
        self.dense_points_colors = dense_colors_rgb  # 保存稠密点云颜色
        
        # 重新绘制整个图表，这样更可靠
        self.redraw_all_points()
    
    def clear_dense_points(self):
        """清除稠密点云显示"""
        self.dense_points_3d = None
        self.dense_points_colors = None  # 清除稠密点云颜色
        self.redraw_all_points()