# 振动加速度数据分析工具 - 批量处理版本
# 作者: StupidMoonlight
# 功能: 读取STS格式的振动数据文件，进行FFT频谱分析并生成可视化图像

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QFileDialog, QGroupBox,
                             QGridLayout, QMessageBox, QAction, QMenuBar, QListWidget,
                             QListWidgetItem, QCheckBox, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
import sys

# 配置matplotlib中文字体，防止中文显示乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'

# 自定义色条颜色映射：黑-绿-青-蓝-紫-红-黄-白
custom_colors = [
    (0.0, 0.0, 0.0),   # 黑
    (0.0, 1.0, 0.0),   # 绿
    (0.0, 1.0, 1.0),   # 青
    (0.0, 0.0, 1.0),   # 蓝
    (0.5, 0.0, 0.5),   # 紫
    (1.0, 0.0, 0.0),   # 红
    (1.0, 1.0, 0.0),   # 黄
    (1.0, 1.0, 1.0),   # 白
]
custom_cmap = LinearSegmentedColormap.from_list('custom', custom_colors, N=256)


def read_sts(path):
    """读取STS二进制文件，解析为浮点数据数组"""
    with open(path, "rb") as f:
        data = f.read()
    return np.array(struct.unpack(f"<{len(data) // 4}f", data))


def generate_plot(data, sample_rate, time_range, freq_range, cross_time, cross_time_window,
                  filter_freq, filter_freq_window, filename="", colorbar_max=None,
                  remove_dc=True, detrend=False, window_type="无", resample_factor=10):
    """
    生成FFT频谱分析图像，包含三个子图：
    1. 上方：指定时间窗口的平均频谱图
    2. 下方：瀑布图（频率-时间-幅值）
    3. 左侧：带通滤波后的时域信号图
    
    参数:
        data: 振动数据数组
        sample_rate: 采样频率 (Hz)
        time_range: 时间范围 (start, end) 秒
        freq_range: 频率范围 (start, end) Hz
        cross_time: 频谱截取时刻 (秒)
        cross_time_window: 频谱截取时间窗口宽度 (秒)
        filter_freq: 带通滤波中心频率 (Hz)
        filter_freq_window: 带通滤波频率范围宽度 (Hz)
        filename: 文件名（用于标题显示）
        colorbar_max: 色条最大值，None表示自适应
        remove_dc: 是否去除直流分量
        detrend: 是否去除趋势
        window_type: 窗函数类型（无/汉宁窗/汉明窗/布莱克曼窗）
        resample_factor: 重采样倍数
    """
    # 根据时间范围截取数据
    start_idx = int(time_range[0] * sample_rate)
    end_idx = int(time_range[1] * sample_rate)
    data = data[start_idx:end_idx]

    # 重采样：降低采样率以加快处理速度
    if resample_factor > 1:
        from scipy.signal import decimate
        data = decimate(data, resample_factor)
        sample_rate = sample_rate / resample_factor

    # 去直流分量：减去均值，消除低频偏移
    if remove_dc:
        data = data - np.mean(data)

    # 去趋势：去除线性趋势，避免影响频谱分析
    if detrend:
        from scipy.signal import detrend as scipy_detrend
        data = scipy_detrend(data)

    # 加窗：应用窗函数减少频谱泄漏
    if window_type != "无":
        if window_type == "汉宁窗":
            window = np.hanning(len(data))
        elif window_type == "汉明窗":
            window = np.hamming(len(data))
        elif window_type == "布莱克曼窗":
            window = np.blackman(len(data))
        data = data * window

    # 计算FFT频谱
    n = len(data)
    fft_result = np.fft.fft(data)
    fft_magnitude = np.abs(fft_result)[:n // 2] * 2 / n
    freq = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]

    # 计算瀑布图数据：使用短时傅里叶变换(STFT)
    window_size = int(sample_rate * 0.5)  # 窗口大小：0.5秒
    overlap = window_size // 2  # 重叠率：50%
    num_windows = (len(data) - window_size) // overlap + 1

    waterfall_data = []
    times = []

    for i in range(num_windows):
        start = i * overlap
        end = start + window_size
        window_data = data[start:end]
        # 对每个窗口进行相同的预处理
        if remove_dc:
            window_data = window_data - np.mean(window_data)
        if detrend:
            from scipy.signal import detrend as scipy_detrend
            window_data = scipy_detrend(window_data)
        if window_type != "无":
            if window_type == "汉宁窗":
                w = np.hanning(len(window_data))
            elif window_type == "汉明窗":
                w = np.hamming(len(window_data))
            elif window_type == "布莱克曼窗":
                w = np.blackman(len(window_data))
            window_data = window_data * w
        # 计算每个窗口的FFT
        window_fft = np.fft.fft(window_data)
        window_mag = np.abs(window_fft)[:window_size // 2] * 2 / window_size
        waterfall_data.append(window_mag)
        times.append(time_range[0] + start / sample_rate)

    waterfall_data = np.array(waterfall_data)
    waterfall_freq = np.fft.fftfreq(window_size, 1 / sample_rate)[:window_size // 2]

    # 根据频率范围截取瀑布图数据
    if freq_range is not None:
        wf_mask = (waterfall_freq >= freq_range[0]) & (waterfall_freq <= freq_range[1])
        waterfall_data = waterfall_data[:, wf_mask]
        waterfall_freq = waterfall_freq[wf_mask]

    # 上方频谱图：计算指定时间窗口内的平均频谱
    time_window = cross_time_window / 2
    cross_indices = []
    for i, t in enumerate(times):
        if abs(t - cross_time) <= time_window:
            cross_indices.append(i)

    if len(cross_indices) > 0:
        cross_spectrum = np.mean(waterfall_data[cross_indices], axis=0)
    else:
        cross_spectrum = waterfall_data[0]

    # 计算主频：在选取的频率范围和时间窗口内，排除10Hz以下的低频干扰
    min_main_freq = max(10, freq_range[0]) if freq_range is not None else 10
    valid_mask = waterfall_freq >= min_main_freq
    valid_spectrum = cross_spectrum.copy()
    valid_spectrum[~valid_mask] = 0
    main_freq_idx = np.argmax(valid_spectrum)
    main_freq = waterfall_freq[main_freq_idx]
    main_amp = valid_spectrum[main_freq_idx]

    # 左侧曲线图：带通滤波后的时域信号
    from scipy.signal import butter, sosfiltfilt
    nyq = 0.5 * sample_rate
    lowcut = filter_freq - filter_freq_window / 2
    highcut = filter_freq + filter_freq_window / 2
    low = lowcut / nyq
    high = highcut / nyq
    low = max(low, 0.001)
    high = min(high, 0.999)
    if low >= high:
        high = low + 0.001
    sos = butter(5, [low, high], btype='band', output='sos')
    filtered_data = sosfiltfilt(sos, data)
    filtered_data = np.nan_to_num(filtered_data, nan=0.0, posinf=0.0, neginf=0.0)
    time_array = np.linspace(time_range[0], time_range[1], len(filtered_data))

    # 单位转换：从g转换为m/s²（1g = 9.81 m/s²）
    g_to_mps2 = 9.81
    cross_spectrum = cross_spectrum * g_to_mps2
    filtered_data = filtered_data * g_to_mps2
    waterfall_data = waterfall_data * g_to_mps2

    # 创建图像布局：使用GridSpec精确控制子图位置和大小
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig,
                  width_ratios=[0.6, 4, 0.1],  # 左图:主图:色条宽度比
                  height_ratios=[0.5, 3],      # 上图:下图高度比
                  hspace=0.05, wspace=0.05)

    ax_left = fig.add_subplot(gs[1, 0])   # 左侧：带通滤波时域图
    ax_top = fig.add_subplot(gs[0, 1])    # 上方：平均频谱图
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_top)  # 下方：瀑布图
    ax_cbar = fig.add_subplot(gs[1, 2])   # 右侧：色条

    # 绘制左侧：带通滤波时域图
    ax_left.plot(filtered_data, time_array, color='black', linewidth=0.8)
    ax_left.set_xlabel("幅值", fontsize=14)
    ax_left.set_ylabel("Time (s)", fontsize=14)
    ax_left.grid(True, linestyle='--', alpha=0.7)
    ax_left.set_ylim(time_range[0], time_range[1])
    # 计算自适应坐标范围，添加5%边距
    filt_min = np.min(filtered_data)
    filt_max = np.max(filtered_data)
    filt_range = filt_max - filt_min
    if filt_range == 0 or np.isnan(filt_range):
        filt_margin = 1.0
    else:
        filt_margin = filt_range * 0.05
    # 反转X轴，使最大值在左侧
    ax_left.set_xlim(filt_max + filt_margin, filt_min - filt_margin)
    ax_left.tick_params(axis='x', labelsize=12, labelrotation=90)
    ax_left.tick_params(axis='y', labelsize=12)
    ax_left.minorticks_on()
    ax_left.tick_params(axis='both', which='minor', length=3)
    ax_left.tick_params(axis='both', which='major', length=6)

    # 绘制上方：指定时间窗口平均频谱图
    ax_top.plot(waterfall_freq, cross_spectrum, color='black', linewidth=0.8)
    ax_top.axvline(x=main_freq, color='red', linestyle='--', linewidth=0.8,
                   label=f'主频: {main_freq:.2f} Hz')
    ax_top.set_title(f"{filename} - Avg at {cross_time}s±{time_window}s", fontsize=16)
    ax_top.set_ylabel("幅值", fontsize=14)
    ax_top.set_xlim(freq_range[0], freq_range[1])
    # 计算自适应坐标范围，添加5%边距
    spec_min = np.min(cross_spectrum)
    spec_max = np.max(cross_spectrum)
    spec_range = spec_max - spec_min
    if spec_range == 0 or np.isnan(spec_range):
        spec_margin = 1.0
    else:
        spec_margin = spec_range * 0.05
    ax_top.set_ylim(spec_min - spec_margin, spec_max + spec_margin)
    ax_top.grid(True, linestyle='--', alpha=0.7)
    ax_top.legend(fontsize=12)
    ax_top.tick_params(axis='both', labelsize=12)
    ax_top.minorticks_on()
    ax_top.tick_params(axis='both', which='minor', length=3)
    ax_top.tick_params(axis='both', which='major', length=6)
    plt.setp(ax_top.get_xticklabels(), visible=False)  # 隐藏X轴标签，与瀑布图对齐

    # 绘制瀑布图：使用对数刻度增强对比度
    waterfall_log = np.log10(waterfall_data + 1e-10)
    log_min = max(np.min(waterfall_log), 0)
    if colorbar_max is not None:
        log_max = np.log10(colorbar_max)
    else:
        log_max = np.max(waterfall_log)
    im = ax_bottom.imshow(waterfall_log, aspect='auto', origin='lower',
                          extent=[waterfall_freq[0], waterfall_freq[-1], times[0], times[-1]],
                          cmap=custom_cmap, interpolation='bilinear',
                          vmin=log_min, vmax=log_max)
    ax_bottom.set_xlabel("Frequency (Hz)", fontsize=14)
    ax_bottom.set_xlim(freq_range[0], freq_range[1])
    ax_bottom.tick_params(axis='both', labelsize=12)
    ax_bottom.minorticks_on()
    ax_bottom.tick_params(axis='both', which='minor', length=3)
    ax_bottom.tick_params(axis='both', which='major', length=6)
    plt.setp(ax_bottom.get_yticklabels(), visible=False)  # 隐藏Y轴标签

    # 用极细红色虚线标注两个剖面位置
    ax_bottom.axhline(y=cross_time, color='red', linestyle='--', linewidth=0.5)  # 频谱剖面
    ax_bottom.axvline(x=filter_freq, color='red', linestyle='--', linewidth=0.5)  # 滤波剖面

    # 绘制色条
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.ax.set_ylabel('幅值', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # 色条标签格式化：将对数刻度转换为真实值显示
    def log_to_real(val, pos):
        real_val = 10 ** val
        if real_val < 0.001:
            return f'{real_val:.1e}'
        elif real_val < 0.1:
            return f'{real_val:.4f}'
        elif real_val < 10:
            return f'{real_val:.3f}'
        else:
            return f'{real_val:.2f}'

    cbar.locator = ticker.MaxNLocator(nbins=10)
    cbar.formatter = ticker.FuncFormatter(log_to_real)
    cbar.update_ticks()

    return fig


class MainWindow(QMainWindow):
    """主窗口类：实现GUI界面和交互逻辑"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("振动加速度数据分析工具")
        self.setGeometry(100, 100, 1600, 900)
        # 设置窗口图标，兼容打包后的路径
        if getattr(sys, 'frozen', False):
            icon_path = os.path.join(sys._MEIPASS, "icon.ico")
        else:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # 设置全局字体
        self.font = QFont()
        self.font.setPointSize(14)
        
        # 初始化数据存储变量
        self.sts_path = ""          # 当前选择的STS文件路径
        self.data = None            # 当前加载的数据
        self.figure = None          # 当前显示的图像
        self.canvas = None          # 图像画布
        self.folder_path = ""       # 当前选择的文件夹路径
        self.file_list = []         # 文件列表
        self.colorbar_max = None    # 色条最大值，None表示自适应
        self.window_type = "无"     # 加窗类型

        # 创建菜单栏
        menubar = self.menuBar()
        
        # 设置菜单：包含清除文件夹、修改色条范围、加窗选项
        settings_menu = menubar.addMenu("设置")
        clear_folder_action = QAction("清除文件夹选择", self)
        clear_folder_action.triggered.connect(self.clear_folder)
        settings_menu.addAction(clear_folder_action)
        
        self.colorbar_action = QAction("修改瀑布图色条范围（默认）", self)
        self.colorbar_action.triggered.connect(self.set_colorbar_max)
        settings_menu.addAction(self.colorbar_action)
        
        self.window_action = QAction("加窗（无）", self)
        self.window_action.triggered.connect(self.set_window_type)
        settings_menu.addAction(self.window_action)
        
        # 帮助菜单：包含关于和使用说明
        help_menu = menubar.addMenu("帮助")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        usage_action = QAction("使用说明", self)
        usage_action.triggered.connect(self.show_usage)
        help_menu.addAction(usage_action)

        # 免责声明菜单
        disclaimer_menu = menubar.addMenu("免责声明")
        disclaimer_action = QAction("查看免责声明", self)
        disclaimer_action.triggered.connect(self.show_disclaimer)
        disclaimer_menu.addAction(disclaimer_action)

        # 创建主布局 - 三列布局
        main_layout = QHBoxLayout()

        # 左侧控制面板 - 参数设置和前处理设置
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(380)

        # 参数设置区域
        param_group = QGroupBox("参数设置")
        param_group.setFont(self.font)
        param_layout = QVBoxLayout()
        param_layout.setSpacing(12)
        
        label_font = QFont()
        label_font.setPointSize(9)
        edit_font = QFont()
        edit_font.setPointSize(11)

        # 时间下限
        time_min_layout = QVBoxLayout()
        label = QLabel("时间下限 (s):")
        label.setFont(label_font)
        time_min_layout.addWidget(label)
        self.time_min = QLineEdit("10")
        self.time_min.setFont(edit_font)
        time_min_layout.addWidget(self.time_min)
        param_layout.addLayout(time_min_layout)

        # 时间上限
        time_max_layout = QVBoxLayout()
        label = QLabel("时间上限 (s):")
        label.setFont(label_font)
        time_max_layout.addWidget(label)
        self.time_max = QLineEdit("25")
        self.time_max.setFont(edit_font)
        time_max_layout.addWidget(self.time_max)
        param_layout.addLayout(time_max_layout)

        # 频率下限
        freq_min_layout = QVBoxLayout()
        label = QLabel("频率下限 (Hz):")
        label.setFont(label_font)
        freq_min_layout.addWidget(label)
        self.freq_min = QLineEdit("0")
        self.freq_min.setFont(edit_font)
        freq_min_layout.addWidget(self.freq_min)
        param_layout.addLayout(freq_min_layout)

        # 频率上限
        freq_max_layout = QVBoxLayout()
        label = QLabel("频率上限 (Hz):")
        label.setFont(label_font)
        freq_max_layout.addWidget(label)
        self.freq_max = QLineEdit("2000")
        self.freq_max.setFont(edit_font)
        freq_max_layout.addWidget(self.freq_max)
        param_layout.addLayout(freq_max_layout)

        # 采样频率
        sample_rate_layout = QVBoxLayout()
        label = QLabel("采样频率 (Hz):")
        label.setFont(label_font)
        sample_rate_layout.addWidget(label)
        self.sample_rate = QLineEdit("51200")
        self.sample_rate.setFont(edit_font)
        sample_rate_layout.addWidget(self.sample_rate)
        param_layout.addLayout(sample_rate_layout)

        # 频谱图截取时刻
        cross_time_layout = QVBoxLayout()
        label = QLabel("频谱截取时刻 (s):")
        label.setFont(label_font)
        cross_time_layout.addWidget(label)
        self.cross_time = QLineEdit("20")
        self.cross_time.setFont(edit_font)
        cross_time_layout.addWidget(self.cross_time)
        param_layout.addLayout(cross_time_layout)

        # 时间范围
        cross_time_window_layout = QVBoxLayout()
        label = QLabel("时间范围 (s):")
        label.setFont(label_font)
        cross_time_window_layout.addWidget(label)
        self.cross_time_window = QLineEdit("0.5")
        self.cross_time_window.setFont(edit_font)
        cross_time_window_layout.addWidget(self.cross_time_window)
        param_layout.addLayout(cross_time_window_layout)

        # 滤波中心频率
        filter_freq_layout = QVBoxLayout()
        label = QLabel("滤波中心频率 (Hz):")
        label.setFont(label_font)
        filter_freq_layout.addWidget(label)
        self.filter_freq = QLineEdit("100")
        self.filter_freq.setFont(edit_font)
        filter_freq_layout.addWidget(self.filter_freq)
        param_layout.addLayout(filter_freq_layout)

        # 频率范围
        filter_freq_window_layout = QVBoxLayout()
        label = QLabel("频率范围 (Hz):")
        label.setFont(label_font)
        filter_freq_window_layout.addWidget(label)
        self.filter_freq_window = QLineEdit("30")
        self.filter_freq_window.setFont(edit_font)
        filter_freq_window_layout.addWidget(self.filter_freq_window)
        param_layout.addLayout(filter_freq_window_layout)

        param_group.setLayout(param_layout)
        control_layout.addWidget(param_group)

        # 前处理设置
        preprocess_group = QGroupBox("前处理设置")
        preprocess_group.setFont(self.font)
        preprocess_layout = QVBoxLayout()
        preprocess_layout.setSpacing(12)

        # 去直流分量
        self.remove_dc = QCheckBox("去直流分量")
        self.remove_dc.setFont(edit_font)
        self.remove_dc.setChecked(True)
        preprocess_layout.addWidget(self.remove_dc)

        # 去趋势
        self.detrend = QCheckBox("去趋势")
        self.detrend.setFont(edit_font)
        self.detrend.setChecked(False)
        preprocess_layout.addWidget(self.detrend)

        # 重采样
        resample_layout = QVBoxLayout()
        resample_label = QLabel("重采样倍数：")
        resample_label.setFont(label_font)
        resample_layout.addWidget(resample_label)
        self.resample_factor = QLineEdit("2")
        self.resample_factor.setFont(edit_font)
        resample_layout.addWidget(self.resample_factor)
        preprocess_layout.addLayout(resample_layout)

        preprocess_group.setLayout(preprocess_layout)
        control_layout.addWidget(preprocess_group)

        # 按钮
        btn_font = QFont()
        btn_font.setPointSize(10)
        self.generate_btn = QPushButton("生成图像")
        self.generate_btn.setFont(btn_font)
        self.generate_btn.setFixedHeight(35)
        self.generate_btn.clicked.connect(self.generate_image)
        control_layout.addWidget(self.generate_btn)

        self.save_btn = QPushButton("保存图像")
        self.save_btn.setFont(btn_font)
        self.save_btn.setFixedHeight(35)
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        control_layout.addWidget(self.save_btn)

        self.save_all_btn = QPushButton("保存所有图像")
        self.save_all_btn.setFont(btn_font)
        self.save_all_btn.setFixedHeight(35)
        self.save_all_btn.clicked.connect(self.save_all_images)
        self.save_all_btn.setEnabled(False)
        control_layout.addWidget(self.save_all_btn)

        control_layout.addStretch()
        main_layout.addWidget(control_panel)

        # 右侧控制面板 - 文件选择和文件列表
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        right_panel.setFixedWidth(380)

        # 文件选择
        file_group = QGroupBox("文件选择")
        file_group.setFont(self.font)
        file_layout = QVBoxLayout()
        
        # 单文件选择
        self.file_label = QLabel("未选择文件")
        file_label_font = QFont()
        file_label_font.setPointSize(10)
        self.file_label.setFont(file_label_font)
        self.file_label.setWordWrap(True)
        self.file_btn = QPushButton("选择STS文件")
        file_btn_font = QFont()
        file_btn_font.setPointSize(10)
        self.file_btn.setFont(file_btn_font)
        self.file_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_btn)
        
        # 分隔线
        from PyQt5.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        file_layout.addWidget(line)
        
        # 文件夹选择
        self.folder_label = QLabel("未选择文件夹")
        self.folder_label.setFont(file_label_font)
        self.folder_label.setWordWrap(True)
        self.folder_btn = QPushButton("选择文件夹")
        folder_btn_font = QFont()
        folder_btn_font.setPointSize(10)
        self.folder_btn.setFont(folder_btn_font)
        self.folder_btn.clicked.connect(self.select_folder)
        file_layout.addWidget(self.folder_label)
        file_layout.addWidget(self.folder_btn)
        
        file_group.setLayout(file_layout)
        right_layout.addWidget(file_group)

        # 文件列表
        file_list_group = QGroupBox("STS文件列表")
        file_list_group.setFont(self.font)
        file_list_layout = QVBoxLayout()
        self.file_list_widget = QListWidget()
        self.file_list_widget.setFont(QFont("Consolas", 10))
        self.file_list_widget.itemClicked.connect(self.on_file_clicked)
        file_list_layout.addWidget(self.file_list_widget)
        file_list_group.setLayout(file_list_layout)
        right_layout.addWidget(file_list_group)

        right_layout.addStretch()
        main_layout.addWidget(right_panel)

        # 右侧图像显示区
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout()
        self.plot_widget.setLayout(self.plot_layout)
        main_layout.addWidget(self.plot_widget)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def select_file(self):
        """选择单个STS文件并读取数据"""
        path, _ = QFileDialog.getOpenFileName(self, "选择STS文件", "", "STS文件 (*.sts)")
        if path:
            self.sts_path = path
            self.file_label.setText(os.path.basename(path))
            try:
                self.data = read_sts(path)
                QMessageBox.information(self, "成功", "文件读取成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"文件读取失败: {str(e)}")

    def select_folder(self):
        """选择包含STS文件的文件夹，并按文件名排序显示"""
        path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if path:
            self.folder_path = path
            self.folder_label.setText(path)
            self.file_list_widget.clear()
            self.file_list = []
            file_pairs = []
            # 遍历文件夹，提取STS文件信息
            for f in os.listdir(path):
                if f.lower().endswith('.sts'):
                    suffix = f.split('_')[-1] if '_' in f else f
                    num = 0
                    # 根据#后的数字排序
                    if '#' in f:
                        try:
                            num_str = f.split('#')[1].split('_')[0]
                            num = int(num_str)
                        except:
                            num = 0
                    file_pairs.append((num, suffix, f))
            # 按#后的数字排序
            file_pairs.sort(key=lambda x: x[0])
            for num, suffix, full_name in file_pairs:
                full_path = os.path.join(path, full_name)
                self.file_list.append(full_path)
                self.file_list_widget.addItem(suffix)
            if not self.file_list:
                QMessageBox.warning(self, "警告", "文件夹中没有STS文件！")
            else:
                QMessageBox.information(self, "成功", f"找到 {len(self.file_list)} 个STS文件！")

    def clear_folder(self):
        """清除文件夹选择和文件列表"""
        self.folder_path = ""
        self.file_list = []
        self.folder_label.setText("未选择文件夹")
        self.file_list_widget.clear()
        self.colorbar_max = None
        self.colorbar_action.setText("修改瀑布图色条范围（默认）")
        self.window_type = "无"
        self.window_action.setText("加窗（无）")

    def set_window_type(self):
        """设置加窗类型，并自动更新图像"""
        from PyQt5.QtWidgets import QInputDialog
        items = ["无", "汉宁窗", "汉明窗", "布莱克曼窗"]
        current_idx = items.index(self.window_type) if self.window_type in items else 0
        item, ok = QInputDialog.getItem(self, "选择加窗类型", "请选择窗函数：", items, current_idx, False)
        if ok:
            self.window_type = item
            self.window_action.setText(f"加窗（{item}）")
            # 自动更新图像
            if self.sts_path and self.data is not None:
                try:
                    time_range = (float(self.time_min.text()), float(self.time_max.text()))
                    freq_range = (float(self.freq_min.text()), float(self.freq_max.text()))
                    sample_rate = int(self.sample_rate.text())
                    cross_time = float(self.cross_time.text())
                    cross_time_window = float(self.cross_time_window.text())
                    filter_freq = float(self.filter_freq.text())
                    filter_freq_window = float(self.filter_freq_window.text())
                    resample_factor = int(self.resample_factor.text())

                    fig = generate_plot(self.data, sample_rate, time_range, freq_range,
                                       cross_time, cross_time_window,
                                       filter_freq, filter_freq_window,
                                       filename=os.path.basename(self.sts_path),
                                       colorbar_max=self.colorbar_max,
                                       remove_dc=self.remove_dc.isChecked(),
                                       detrend=self.detrend.isChecked(),
                                       window_type=self.window_type,
                                       resample_factor=resample_factor)

                    for i in reversed(range(self.plot_layout.count())):
                        self.plot_layout.itemAt(i).widget().deleteLater()

                    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
                    self.canvas = FigureCanvasQTAgg(fig)
                    self.plot_layout.addWidget(self.canvas)
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"更新图像失败：{str(e)}")

    def set_colorbar_max(self):
        from PyQt5.QtWidgets import QInputDialog
        if self.colorbar_max is None:
            default_val = ""
        else:
            default_val = str(self.colorbar_max)
        text, ok = QInputDialog.getText(self, "设置色条范围", "请输入色条最大值（留空表示自适应）：", text=default_val)
        if ok:
            if text.strip() == "":
                self.colorbar_max = None
                self.colorbar_action.setText("修改瀑布图色条范围（默认）")
            else:
                try:
                    val = float(text)
                    self.colorbar_max = val
                    self.colorbar_action.setText(f"修改瀑布图色条范围（{val:g}）")
                except ValueError:
                    QMessageBox.warning(self, "警告", "请输入有效的数字！")
                    return
            # 自动更新图像
            if self.sts_path and self.data is not None:
                try:
                    time_range = (float(self.time_min.text()), float(self.time_max.text()))
                    freq_range = (float(self.freq_min.text()), float(self.freq_max.text()))
                    sample_rate = int(self.sample_rate.text())
                    cross_time = float(self.cross_time.text())
                    cross_time_window = float(self.cross_time_window.text())
                    filter_freq = float(self.filter_freq.text())
                    filter_freq_window = float(self.filter_freq_window.text())

                    fig = generate_plot(self.data, sample_rate, time_range, freq_range,
                                       cross_time, cross_time_window,
                                       filter_freq, filter_freq_window,
                                       filename=os.path.basename(self.sts_path),
                                       colorbar_max=self.colorbar_max,
                                       remove_dc=self.remove_dc.isChecked(),
                                       detrend=self.detrend.isChecked(),
                                       window_type=self.window_type,
                                       resample_factor=int(self.resample_factor.text()))

                    for i in reversed(range(self.plot_layout.count())):
                        self.plot_layout.itemAt(i).widget().deleteLater()

                    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
                    self.canvas = FigureCanvasQTAgg(fig)
                    self.plot_layout.addWidget(self.canvas)
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"更新图像失败：{str(e)}")

    def on_file_clicked(self, item):
        idx = self.file_list_widget.row(item)
        if idx < len(self.file_list):
            file_path = self.file_list[idx]
            try:
                data = read_sts(file_path)
                self.sts_path = file_path
                self.data = data
                time_range = (float(self.time_min.text()), float(self.time_max.text()))
                freq_range = (float(self.freq_min.text()), float(self.freq_max.text()))
                sample_rate = int(self.sample_rate.text())
                cross_time = float(self.cross_time.text())
                cross_time_window = float(self.cross_time_window.text())
                filter_freq = float(self.filter_freq.text())
                filter_freq_window = float(self.filter_freq_window.text())

                fig = generate_plot(data, sample_rate, time_range, freq_range,
                                   cross_time, cross_time_window,
                                   filter_freq, filter_freq_window,
                                   filename=os.path.basename(file_path),
                                   colorbar_max=self.colorbar_max,
                                       remove_dc=self.remove_dc.isChecked(),
                                       detrend=self.detrend.isChecked(),
                                       window_type=self.window_type,
                                       resample_factor=int(self.resample_factor.text()))

                for i in reversed(range(self.plot_layout.count())):
                    self.plot_layout.itemAt(i).widget().deleteLater()

                self.figure = fig
                self.canvas = FigureCanvas(self.figure)
                toolbar = NavigationToolbar(self.canvas, self)
                self.plot_layout.addWidget(toolbar)
                self.plot_layout.addWidget(self.canvas)
                self.canvas.draw()
                self.save_btn.setEnabled(True)
                if self.folder_path and self.file_list:
                    self.save_all_btn.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"图像生成失败: {str(e)}")

    def generate_image(self):
        if not self.sts_path:
            QMessageBox.warning(self, "警告", "请先选择STS文件！")
            return

        try:
            time_range = (float(self.time_min.text()), float(self.time_max.text()))
            freq_range = (float(self.freq_min.text()), float(self.freq_max.text()))
            sample_rate = int(self.sample_rate.text())
            cross_time = float(self.cross_time.text())
            cross_time_window = float(self.cross_time_window.text())
            filter_freq = float(self.filter_freq.text())
            filter_freq_window = float(self.filter_freq_window.text())
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的数值！")
            return

        if time_range[0] >= time_range[1]:
            QMessageBox.warning(self, "警告", "时间下限必须小于时间上限！")
            return

        if freq_range[0] >= freq_range[1]:
            QMessageBox.warning(self, "警告", "频率下限必须小于频率上限！")
            return

        try:
            self.figure = generate_plot(self.data, sample_rate, time_range, freq_range,
                                       cross_time, cross_time_window,
                                       filter_freq, filter_freq_window,
                                       filename=os.path.basename(self.sts_path),
                                       colorbar_max=self.colorbar_max,
                                       remove_dc=self.remove_dc.isChecked(),
                                       detrend=self.detrend.isChecked(),
                                       window_type=self.window_type,
                                       resample_factor=int(self.resample_factor.text()))

            # 清除旧图像
            for i in reversed(range(self.plot_layout.count())):
                self.plot_layout.itemAt(i).widget().deleteLater()

            # 添加新图像
            self.canvas = FigureCanvas(self.figure)
            toolbar = NavigationToolbar(self.canvas, self)
            self.plot_layout.addWidget(toolbar)
            self.plot_layout.addWidget(self.canvas)
            self.canvas.draw()

            self.save_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"图像生成失败: {str(e)}")

    def save_image(self):
        if not self.figure:
            return

        path, _ = QFileDialog.getSaveFileName(self, "保存图像", "", "PNG图像 (*.png);;JPEG图像 (*.jpg)")
        if path:
            try:
                self.figure.savefig(path, dpi=150, bbox_inches='tight')
                QMessageBox.information(self, "成功", "图像保存成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def save_all_images(self):
        if not self.folder_path or not self.file_list:
            QMessageBox.warning(self, "警告", "请先选择包含STS文件的文件夹！")
            return

        try:
            time_range = (float(self.time_min.text()), float(self.time_max.text()))
            freq_range = (float(self.freq_min.text()), float(self.freq_max.text()))
            sample_rate = int(self.sample_rate.text())
            cross_time = float(self.cross_time.text())
            cross_time_window = float(self.cross_time_window.text())
            filter_freq = float(self.filter_freq.text())
            filter_freq_window = float(self.filter_freq_window.text())
            resample_factor = int(self.resample_factor.text())
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的数值！")
            return

        # 创建输出文件夹
        output_folder = os.path.join(self.folder_path, "图像")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        success_count = 0
        fail_count = 0
        fail_files = []

        from PyQt5.QtWidgets import QProgressDialog
        progress = QProgressDialog("正在生成图像...", "取消", 0, len(self.file_list), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        for i, file_path in enumerate(self.file_list):
            if progress.wasCanceled():
                break

            progress.setValue(i)
            progress.setLabelText(f"正在处理: {os.path.basename(file_path)}")

            try:
                data = read_sts(file_path)
                fig = generate_plot(data, sample_rate, time_range, freq_range,
                                   cross_time, cross_time_window,
                                   filter_freq, filter_freq_window,
                                   filename=os.path.basename(file_path),
                                   colorbar_max=self.colorbar_max,
                                   remove_dc=self.remove_dc.isChecked(),
                                   detrend=self.detrend.isChecked(),
                                   window_type=self.window_type,
                                   resample_factor=resample_factor)

                # 使用原文件名（去掉.sts后缀）作为图像文件名
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(output_folder, f"{base_name}.png")
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                success_count += 1
            except Exception as e:
                fail_count += 1
                fail_files.append(os.path.basename(file_path))

        progress.setValue(len(self.file_list))

        msg = f"批量保存完成！\n\n成功: {success_count} 个\n失败: {fail_count} 个\n保存位置: {output_folder}"
        if fail_files:
            msg += f"\n\n失败文件:\n" + "\n".join(fail_files[:10])
            if len(fail_files) > 10:
                msg += f"\n...及其他{len(fail_files)-10}个文件"

        if success_count > 0:
            QMessageBox.information(self, "完成", msg)
            # 打开源数据文件夹
            import subprocess
            folder_path = os.path.normpath(self.folder_path)
            subprocess.Popen(f'explorer "{folder_path}"')
        else:
            QMessageBox.critical(self, "错误", msg)

    def show_about(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("关于")
        msg.setTextFormat(Qt.RichText)
        msg.setText(
            "振动加速度数据分析工具 v1.0<br><br>"
            "作者: StupidMoonlight<br>"
            "GitHub: <a href='https://github.com/StupidMoonlight'>https://github.com/StupidMoonlight</a><br><br>"
            "用于振动加速度信号的FFT频谱分析和可视化。")
        msg.setTextInteractionFlags(Qt.TextBrowserInteraction)
        msg.exec_()

    def show_usage(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
        from PyQt5.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("使用说明")
        dialog.resize(700, 500)
        
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml(
            "<h3>【使用说明】</h3>"
            "<h4>1. 选择数据文件：</h4>"
            "<ul>"
            "<li>点击<b>\"选择STS文件\"</b>按钮，选择单个要分析的.sts振动数据文件。</li>"
            "<li>点击<b>\"选择文件夹\"</b>按钮，选择包含多个.sts文件的文件夹，程序会自动列出所有STS文件，点击文件即可预览对应图像。</li>"
            "</ul>"
            "<h4>2. 设置参数：</h4>"
            "<ul>"
            "<li><b>时间上下限</b>：选择要分析的时间段范围（单位：秒）</li>"
            "<li><b>频率上下限</b>：设置显示的频率范围（单位：Hz，默认0-2000）</li>"
            "<li><b>采样频率</b>：数据的采样频率（单位：Hz，默认51200）</li>"
            "<li><b>频谱截取时刻</b>：上方频谱图的截取时间点（单位：秒）</li>"
            "<li><b>时间范围</b>：频谱截取的时间窗口宽度，程序识别为±一半（如填0.5，则截取时刻±0.25秒）</li>"
            "<li><b>滤波中心频率</b>：左侧曲线图的带通滤波中心频率（单位：Hz）</li>"
            "<li><b>频率范围</b>：带通滤波的频率宽度，程序识别为±一半（如填30，则中心频率±15Hz）</li>"
            "</ul>"
            "<h4>3. 前处理设置：</h4>"
            "<ul>"
            "<li><b>去直流分量</b>：去除信号的直流分量（默认开启）</li>"
            "<li><b>去趋势</b>：去除信号的线性趋势（默认关闭）</li>"
            "<li><b>重采样倍数</b>：降低采样率以加快处理速度（默认2倍）</li>"
            "<li><b>加窗</b>：可在\"设置\"菜单中选择窗函数类型（默认无）</li>"
            "</ul>"
            "<h4>4. 生成与保存图像：</h4>"
            "<ul>"
            "<li>点击<b>\"生成图像\"</b>按钮，右侧将显示分析结果图像。</li>"
            "<li>点击<b>\"保存图像\"</b>按钮，可将当前图像保存为PNG或JPEG格式。</li>"
            "<li>点击<b>\"保存所有图像\"</b>按钮，可批量生成文件夹中所有STS文件的图像，自动保存在源文件夹下的\"图像\"子文件夹中，完成后自动弹出该文件夹。</li>"
            "</ul>"
            "<h4>5. 设置菜单：</h4>"
            "<ul>"
            "<li><b>清除文件夹选择</b>：清空已选择的文件夹和文件列表</li>"
            "<li><b>修改瀑布图色条范围</b>：设置色条的最大值（默认自适应）</li>"
            "<li><b>加窗</b>：选择窗函数类型（无/汉宁窗/汉明窗/布莱克曼窗）</li>"
            "</ul>"
            "<h3>【图像说明】</h3>"
            "<ul>"
            "<li><b>上方</b>：指定时刻的频域剖面图（FFT频谱），红色虚线标注主频</li>"
            "<li><b>中间</b>：瀑布图（频率-时间-幅值），两条红色虚线标注剖面位置</li>"
            "<li><b>左侧</b>：带通滤波后的时域信号</li>"
            "<li><b>右侧</b>：色条（对数刻度）</li>"
            "</ul>"
        )
        layout.addWidget(text_edit)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def show_disclaimer(self):
        QMessageBox.warning(self, "免责声明",
            "【免责声明】\n\n"
            "1. 本软件仅供学术研究和教学用途，不用于任何商业目的。\n\n"
            "2. 本软件按\"原样\"提供，不作任何明示或暗示的保证，包括但不限于"
            "适销性、特定用途的适用性和非侵权性。\n\n"
            "3. 在任何情况下，作者或版权持有人均不对因使用本软件而产生的任何"
            "直接、间接、附带、特殊、惩罚性或后果性损害承担责任。\n\n"
            "4. 用户应自行承担使用本软件的风险。作者不对因使用本软件而导致的"
            "数据丢失、分析错误或任何其他损失负责。\n\n"
            "5. 本软件的分析结果仅供参考，不应作为工程决策的唯一依据。"
            "对于涉及安全的工程应用，请使用经过认证的专业分析工具。")


if __name__ == "__main__":
    import sys
    import ctypes
    app_id = "StupidMoonlight.VibrationAnalyzer.v1.0"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    app = QApplication(sys.argv)
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(base_path, "icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
