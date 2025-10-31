#!/usr/bin/env python3
import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

class VideoWidget(QLabel):
    def __init__(self, title):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setStyleSheet("border: 1px solid gray;")
        self.setAlignment(Qt.AlignCenter)
        self.setText(f"{title}\n等待相机...")
        self.title = title
        self.matches = None
        self.current_frame = None
        
    def update_frame(self, frame):
        self.current_frame = frame.copy()
        self.draw_frame()
        
    def update_matches(self, matches, camera_id):
        if (camera_id == 0 and "左" in self.title) or (camera_id == 1 and "右" in self.title):
            self.matches = matches
            self.draw_frame()
            
    def draw_frame(self):
        if self.current_frame is None:
            return
            
        frame = self.current_frame.copy()
        
        # Draw matches if available
        if self.matches is not None and len(self.matches) > 0:
            for i, (x, y) in enumerate(self.matches):
                # Use different colors based on point index
                color = (0, 255, 0) if i % 2 == 0 else (255, 0, 0)
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
                
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit widget
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)