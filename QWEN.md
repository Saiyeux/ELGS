# ELGS (EfficientLoFTR + Gaussian Splatting) - 项目上下文

## 项目概述

ELGS是一个集成立体视觉与神经渲染的实时3D重建系统。它结合了EfficientLoFTR特征匹配和3D Gaussian Splatting稠密重建技术，旨在提供高质量、实时的3D场景重建能力。

### 核心技术栈

- **前端**: PyQt5 GUI框架
- **计算机视觉**: OpenCV, EfficientLoFTR
- **深度学习**: PyTorch, CUDA
- **3D渲染**: 3D Gaussian Splatting, matplotlib
- **数值计算**: NumPy

### 核心特性

- 实时双目立体匹配
- 固定世界坐标系3D重建
- 多尺度稠密点云生成
- 防闪烁3D可视化
- 模块化可扩展架构

## 项目结构

```
ELGS/
├── EDGS/                 # 核心应用代码
│   ├── core/             # 核心控制逻辑
│   ├── model/            # 数据模型和处理线程
│   ├── control/          # UI控件和控制器
│   └── main.py           # 应用程序入口
├── config/               # 配置管理
├── thirdparty/           # 第三方库和模型
├── assets/               # 资源文件
├── notebooks/            # Jupyter笔记本
├── output/               # 输出目录
├── run.py                # 启动脚本
├── environment.yaml      # Conda环境配置
└── ELGS_DETAILED_DESIGN_DOCUMENT.md  # 详细设计文档
```

## 构建和运行

### 环境要求

- **操作系统**: Linux (推荐Ubuntu 18.04+), Windows, 或 macOS
- **Python**: 3.8-3.10
- **CUDA**: 11.6+ (推荐)
- **GPU**: NVIDIA GPU with at least 6GB VRAM (推荐RTX 3060+)

### 安装步骤

1. **创建Conda环境**:
   ```bash
   conda env create -f environment.yaml
   conda activate ELGS
   ```

2. **安装依赖**:
   ```bash
   pip install -r requirements.txt  # 如果有requirements.txt文件
   ```
   或者使用启动脚本自动检查和安装:
   ```bash
   python run.py
   ```

3. **编译CUDA扩展** (如果需要):
   ```bash
   cd thirdparty/gaussian-splatting/submodules
   pip install diff-gaussian-rasterization/
   pip install simple-knn/
   ```

4. **下载预训练模型**:
   从EfficientLoFTR官方仓库下载`eloftr_outdoor.ckpt`，并放置于`thirdparty/EfficientLoFTR/weights/`目录下。

### 启动应用

```bash
python run.py
```

这将检查依赖、权重文件和GPU支持，然后启动Qt GUI应用程序。

## 核心模块

### 1. EDGS/core/

- **main_window.py**: 主窗口控制器，管理所有线程和UI组件。

### 2. EDGS/model/

- **camera_thread.py**: 独立的相机数据采集线程。
- **matching_thread.py**: EfficientLoFTR特征匹配和3D重建线程。
- **filter_thread.py**: 匹配结果的噪声过滤线程。
- **gaussian_thread.py**: 3D Gaussian Splatting稠密重建线程。

### 3. EDGS/control/

- **point3d_widget.py**: 3D点云可视化控件。
- **video_widget.py**: 实时视频流显示控件。
- **config_dialog.py**: 参数配置对话框。

### 4. config/

- **config_manager.py**: 统一配置管理器。
- **gaussian_config.py**: 高斯重建参数配置。
- **loftr_config.py**: 特征匹配参数配置。

## 开发约定

### 代码结构

- 项目遵循模块化设计，分为`core`, `model`, `control`等目录。
- 每个线程负责独立的任务，通过信号槽机制进行通信。
- UI组件与数据处理逻辑分离，便于维护和扩展。

### 配置管理

- 所有参数通过`ConfigManager`统一管理。
- 配置文件为JSON格式，便于读写和版本控制。
- UI配置与内部参数分离，通过专门的方法进行转换。

### 多线程

- 使用PyQt的`QThread`进行多线程处理。
- 线程间通过`pyqtSignal`进行安全通信。
- 避免在非主线程中直接操作UI组件。

### 3D可视化

- 使用matplotlib进行3D点云可视化。
- 通过统一的坐标变换函数确保稀疏点云和稠密点云在相同坐标系下显示。
- 实现了防闪烁机制，固定坐标轴范围，避免频繁重绘导致的视觉不稳定。

## 调试与监控

- 系统通过日志记录关键操作和错误信息。
- UI界面提供实时性能监控和状态显示。
- 通过配置对话框可以实时调整参数并观察效果。

## 扩展性

- 系统设计了清晰的接口，便于集成新的算法模块。
- 配置系统支持扩展新的参数类型。
- 可视化组件支持添加新的渲染方式。