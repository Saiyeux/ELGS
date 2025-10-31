#!/usr/bin/env python3
"""
ELGS Qt GUI 启动脚本
检查依赖并启动Qt界面应用程序
"""
import os
import sys
import subprocess
import importlib.util
from pathlib import Path

# 修复Qt平台插件冲突问题
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
# 禁用OpenCV的Qt后端
os.environ['OPENCV_VIDEOIO_DEBUG'] = '1'
# 设置显示相关环境变量
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

def check_package(package_name, import_name=None):
    """检查Python包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        return spec is not None
    except ImportError:
        return False

def install_package(package_name):
    """安装Python包"""
    print(f"正在安装 {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ {package_name} 安装失败")
        return False

def check_dependencies():
    """检查并安装所需依赖"""
    dependencies = [
        ("PyQt5", "PyQt5"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("open3d", "open3d"),
    ]
    
    missing_packages = []
    
    print("检查依赖包...")
    for package, import_name in dependencies:
        if check_package(package, import_name):
            print(f"✓ {package} 已安装")
        else:
            print(f"✗ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n需要安装 {len(missing_packages)} 个包: {', '.join(missing_packages)}")
        
        for package in missing_packages:
            if not install_package(package):
                print(f"无法安装 {package}, 请手动安装")
                return False
                
        print("\n所有依赖已安装完成!")
    else:
        print("\n所有依赖已满足!")
    
    return True

def check_eloftr_weights():
    """检查EfficientLoFTR权重文件"""
    weights_path = Path("thirdparty/EfficientLoFTR/weights/eloftr_outdoor.ckpt")
    
    if weights_path.exists():
        print(f"✓ EfficientLoFTR权重文件存在: {weights_path}")
        return True
    else:
        print(f"✗ EfficientLoFTR权重文件不存在: {weights_path}")
        print("请从以下链接下载权重文件:")
        print("https://github.com/zju3dv/EfficientLoFTR/releases")
        print("并放置到 thirdparty/EfficientLoFTR/weights/ 目录下")
        return False

def check_gpu_support():
    """检查GPU支持"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ 检测到GPU支持: {gpu_count} 个GPU")
            print(f"  主GPU: {gpu_name}")
            return True
        else:
            print("! 未检测到GPU，将使用CPU模式 (性能较慢)")
            return True
    except Exception as e:
        print(f"! GPU检查出错: {e}")
        return True

def main():
    """主函数"""
    print("="*50)
    print("ELGS Qt GUI 启动检查")
    print("="*50)
    
    # 检查工作目录
    if not Path("EDGS").exists():
        print("✗ 找不到 EDGS 包")
        print("请确保在ELGS项目根目录下运行此脚本")
        return 1
    
    # 检查依赖
    if not check_dependencies():
        print("依赖检查失败，无法启动GUI")
        return 1
    
    # 检查权重文件
    if not check_eloftr_weights():
        print("权重文件缺失，功能可能受限")
    
    # 检查GPU支持
    check_gpu_support()
    
    # 创建输出目录
    Path("output").mkdir(exist_ok=True)
    print("✓ 输出目录已准备")
    
    print("\n" + "="*50)
    print("启动ELGS Qt GUI...")
    print("="*50)
    
    try:
        # 启动GUI应用程序
        from EDGS.main import main as gui_main
        gui_main()
    except KeyboardInterrupt:
        print("\n用户中断程序")
        return 0
    except Exception as e:
        print(f"启动GUI时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)