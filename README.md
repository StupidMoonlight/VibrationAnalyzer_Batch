# VibrationAnalyzer / 振动加速度数据分析工具

## English

A GUI-based tool for vibration acceleration data analysis with FFT spectrum analysis and waterfall plot visualization.

### Features
- **FFT Spectrum Analysis**: Convert time-domain vibration signals to frequency domain
- **Waterfall Plot**: Visualize frequency changes over time with color-coded amplitude
- **Bandpass Filtering**: Extract specific frequency components from vibration signals
- **Batch Processing**: Process multiple STS files automatically
- **Data Preprocessing**: DC removal, detrending, windowing, and resampling
- **Customizable Parameters**: Time/frequency range, sampling rate, colorbar settings

### Usage
1. Select an STS file or folder containing multiple STS files
2. Set analysis parameters (time range, frequency range, sampling rate, etc.)
3. Click "Generate Image" to view the analysis result
4. Click "Save All Images" to batch process all files

### Requirements
- Python 3.7+
- PyQt5, numpy, scipy, matplotlib


---

## 中文

基于PyQt5的振动加速度数据分析工具，支持FFT频谱分析和瀑布图可视化。

### 功能特性
- **FFT频谱分析**：将时域振动信号转换为频域信号
- **瀑布图**：通过颜色编码展示频率随时间的变化
- **带通滤波**：提取特定频率范围的振动信号
- **批量处理**：自动处理多个STS文件
- **数据预处理**：去直流分量、去趋势、加窗、重采样
- **参数自定义**：时间/频率范围、采样频率、色条设置等

### 使用方法
1. 选择单个STS文件或包含多个STS文件的文件夹
2. 设置分析参数（时间范围、频率范围、采样频率等）
3. 点击"生成图像"查看分析结果
4. 点击"保存所有图像"批量处理所有文件

### 依赖环境
- Python 3.7+
- PyQt5, numpy, scipy, matplotlib
