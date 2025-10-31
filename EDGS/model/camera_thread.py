#!/usr/bin/env python3
import cv2
import numpy as np
import threading
import time
from PyQt5.QtCore import QThread, pyqtSignal

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, int)
    
    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        self.skip_frames = 0
        self.frame_count = 0
        
    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        
        # 设置摄像头参数为1080p
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # 统一曝光设置，确保两个摄像头亮度一致
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 自动曝光模式
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)     # 统一亮度设置
        self.cap.set(cv2.CAP_PROP_CONTRAST, 128)     # 统一对比度
        self.cap.set(cv2.CAP_PROP_SATURATION, 128)   # 统一饱和度
        self.cap.set(cv2.CAP_PROP_GAIN, 100)         # 统一增益
        
        # 等待摄像头初始化
        import time
        time.sleep(0.5)
        
        if not self.cap.isOpened():
            return False
            
        # 尝试读取一帧来验证
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            return False
            
        self.running = True
        self.start()
        return True
        
    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()
        
    def set_skip_frames(self, skip):
        self.skip_frames = skip
        
    def run(self):
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                if self.frame_count % (self.skip_frames + 1) == 0:
                    self.frame_ready.emit(frame, self.camera_id)
                self.frame_count += 1
            time.sleep(0.033)  # ~30 FPS