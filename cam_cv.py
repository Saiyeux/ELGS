import cv2
import sys

def get_camera_info(cap, cam_id):
    """获取摄像头详细信息"""
    info = {}
    info['camera_id'] = cam_id
    
    # 基本属性
    info['width'] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    info['height'] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    info['fps'] = cap.get(cv2.CAP_PROP_FPS)
    
    # 编码格式
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    info['fourcc'] = ''.join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
    
    # 曝光和图像参数
    info['brightness'] = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    info['contrast'] = cap.get(cv2.CAP_PROP_CONTRAST)
    info['saturation'] = cap.get(cv2.CAP_PROP_SATURATION)
    info['hue'] = cap.get(cv2.CAP_PROP_HUE)
    info['gain'] = cap.get(cv2.CAP_PROP_GAIN)
    info['exposure'] = cap.get(cv2.CAP_PROP_EXPOSURE)
    info['auto_exposure'] = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    
    # 白平衡
    info['white_balance'] = cap.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)
    info['temperature'] = cap.get(cv2.CAP_PROP_TEMPERATURE)
    
    # 缓冲区信息
    info['buffer_size'] = cap.get(cv2.CAP_PROP_BUFFERSIZE)
    
    return info

def print_camera_info(info):
    """打印摄像头信息"""
    print(f"\n=== 摄像头 {info['camera_id']} 信息 ===")
    print(f"分辨率: {int(info['width'])}x{int(info['height'])}")
    print(f"帧率: {info['fps']:.2f} FPS")
    print(f"编码格式: {info['fourcc']}")
    print(f"亮度: {info['brightness']:.0f}")
    print(f"对比度: {info['contrast']:.0f}")
    print(f"饱和度: {info['saturation']:.0f}")
    print(f"色调: {info['hue']:.0f}")
    print(f"增益: {info['gain']:.0f}")
    print(f"曝光: {info['exposure']:.0f}")
    print(f"自动曝光: {info['auto_exposure']:.0f}")
    print(f"白平衡: {info['white_balance']:.0f}")
    print(f"色温: {info['temperature']:.0f}")
    print(f"缓冲区大小: {info['buffer_size']:.0f}")
    print("=" * 30)

def main():
    # 摄像头ID列表
    camera_ids = [0, 1, 2, 3, 4, 5]
    
    # 存储成功打开的摄像头
    cameras = []
    
    # 遍历摄像头ID列表，尝试打开每个摄像头
    for cam_id in camera_ids:
        cap = cv2.VideoCapture(cam_id)
        
        # 设置摄像头参数为1080p
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # 统一曝光设置，确保两个摄像头亮度一致
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 自动曝光模式
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)     # 统一亮度设置
        cap.set(cv2.CAP_PROP_CONTRAST, 128)     # 统一对比度
        cap.set(cv2.CAP_PROP_SATURATION, 128)   # 统一饱和度
        cap.set(cv2.CAP_PROP_GAIN, 100)         # 统一增益
        
        # 等待摄像头初始化
        import time
        time.sleep(0.5)
        
        if cap.isOpened():
            # 尝试读取一帧来验证
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                cameras.append((cam_id, cap))
                print(f"成功打开摄像头 {cam_id}")
                
                # 获取并显示摄像头详细信息
                info = get_camera_info(cap, cam_id)
                print_camera_info(info)
            else:
                print(f"摄像头 {cam_id} 打开但无法读取帧")
                cap.release()
        else:
            print(f"无法打开摄像头 {cam_id}")
            cap.release()
    
    # 检查是否有可用的摄像头
    if not cameras:
        print("Error: 没有可用的摄像头")
        sys.exit(1)
    
    try:
        while True:
            # 遍历所有打开的摄像头
            for cam_id, cap in cameras:
                # 读取一帧
                ret, frame = cap.read()
                
                # 检查帧是否读取成功
                if ret:
                    # 显示帧，每个摄像头使用不同的窗口名称
                    cv2.imshow(f'Camera {cam_id}', frame)
                else:
                    print(f"Error: 无法从摄像头 {cam_id} 读取帧")
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # 释放所有摄像头资源
        for cam_id, cap in cameras:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()