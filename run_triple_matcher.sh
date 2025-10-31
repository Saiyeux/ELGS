#!/bin/bash

echo "ELGS 三路匹配系统启动脚本"
echo "========================================"

# 检查环境
echo "正在检查运行环境..."

# 检查Python
if ! command -v python &> /dev/null; then
    echo "❌ 错误: Python未找到"
    exit 1
fi
echo "✅ Python: $(python --version)"

# 检查CUDA
echo "正在检查CUDA环境..."
python -c "
import torch
if torch.cuda.is_available():
    print('✅ CUDA可用: GPU数量=' + str(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        print('   设备{}: {}'.format(i, torch.cuda.get_device_name(i)))
else:
    print('⚠️  CUDA不可用，将使用CPU（性能较慢）')
"

# 检查EfficientLoFTR权重
echo "正在检查EfficientLoFTR权重文件..."
if [ -f "thirdparty/EfficientLoFTR/weights/eloftr_outdoor.ckpt" ]; then
    echo "✅ EfficientLoFTR权重文件存在"
else
    echo "❌ 错误: EfficientLoFTR权重文件不存在"
    echo "请确保已下载权重文件到: thirdparty/EfficientLoFTR/weights/eloftr_outdoor.ckpt"
    exit 1
fi

# 检查依赖库
echo "正在检查依赖库..."
python -c "
import sys
sys.path.append('thirdparty/EfficientLoFTR')
sys.path.append('thirdparty/EfficientLoFTR/src')
try:
    from PyQt5.QtWidgets import QApplication
    print('✅ PyQt5可用')
except ImportError:
    print('❌ 错误: PyQt5未安装')
    sys.exit(1)

try:
    import cv2
    print('✅ OpenCV可用: ' + cv2.__version__)
except ImportError:
    print('❌ 错误: OpenCV未安装')
    sys.exit(1)
    
try:
    from src.loftr import LoFTR
    print('✅ EfficientLoFTR模块可用')
except ImportError as e:
    print('❌ 错误: EfficientLoFTR模块导入失败: ' + str(e))
    sys.exit(1)

try:
    import pynvml
    print('✅ pynvml可用（可显示详细GPU信息）')
except ImportError:
    print('⚠️  pynvml不可用，将使用PyTorch备用方案')
"

echo "========================================"
echo "🚀 正在启动三路匹配系统..."
echo ""

# 解析启动参数
ARGS="$@"

# 如果没有参数，使用默认配置
if [ $# -eq 0 ]; then
    echo "使用默认配置启动..."
    python triple_matcher.py
else
    echo "使用自定义参数启动: $ARGS"
    python triple_matcher.py $ARGS
fi