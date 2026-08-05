"""
信号处理模块 - 包含STS文件读取和FFT频谱分析功能
"""
import os
import struct
import numpy as np
from scipy.signal import decimate, detrend as scipy_detrend, butter, sosfiltfilt
from config import custom_cmap


def read_sts(path):
    """读取STS二进制文件，解析为浮点数据数组"""
    with open(path, "rb") as f:
        data = f.read()
    return np.array(struct.unpack(f"<{len(data) // 4}f", data))


def read_tsp_sensor_info(tsp_path):
    """
    读取TSP文件中的传感器名称和工程单位

    参数:
        tsp_path: TSP文件路径

    返回:
        (sensor_name, eu): 传感器名称和工程单位，读取失败返回(None, None)
    """
    try:
        with open(tsp_path, 'r', encoding='gbk', errors='ignore') as f:
            lines = f.readlines()

        sensor_name = None
        eu = None

        # 方法1: 从第二行提取中文名称（优先）
        if len(lines) >= 2:
            second_line = lines[1].strip()
            if second_line:
                sensor_name = second_line

        # 方法2: 从ChNodeName中提取（仅当方法1未获取到名称时作为备选）
        for line in lines:
            if line.startswith('ChNodeName='):
                if not sensor_name:
                    ch_node_name = line.strip().split('=', 1)[1]
                    if '_' in ch_node_name:
                        sensor_name = ch_node_name.split('_', 1)[1]
                    else:
                        sensor_name = ch_node_name
            elif line.startswith('ChNodeEu='):
                eu = line.strip().split('=', 1)[1]

        return sensor_name, eu
    except Exception:
        return None, None


def get_sensor_info(sts_path):
    """
    获取STS文件对应的传感器名称和工程单位

    参数:
        sts_path: STS文件路径

    返回:
        (sensor_name, eu): 传感器名称和工程单位，找不到TSP文件时eu为None
    """
    tsp_path = sts_path.replace('.sts', '.tsp').replace('.STS', '.TSP')

    if os.path.exists(tsp_path):
        sensor_name, eu = read_tsp_sensor_info(tsp_path)
        if sensor_name:
            return sensor_name, eu

    # 如果TSP文件不存在或读取失败，从STS文件名提取
    filename = os.path.basename(sts_path)
    if '_' in filename:
        return filename.split('_')[-1].replace('.sts', '').replace('.STS', ''), None
    return filename.replace('.sts', '').replace('.STS', ''), None


def get_sensor_name(sts_path):
    """
    获取STS文件对应的传感器名称（向后兼容）

    参数:
        sts_path: STS文件路径

    返回:
        sensor_name: 传感器名称，如果找不到TSP文件则返回STS文件名后缀
    """
    name, _ = get_sensor_info(sts_path)
    return name


def generate_plot(data, sample_rate, time_range, freq_range, cross_time, cross_time_window,
                  filter_freq, filter_freq_window, filename="", colorbar_max=None,
                  colorbar_min=None, remove_dc=True, detrend=False, window_type="无", resample_factor=10,
                  min_freq_for_main=10):
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
        colorbar_min: 色条最小值，None表示自适应
        remove_dc: 是否去除直流分量
        detrend: 是否去除趋势
        window_type: 窗函数类型（无/汉宁窗/汉明窗/布莱克曼窗）
        resample_factor: 重采样倍数
        min_freq_for_main: 主频搜索起始频率（Hz）
    """
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.ticker as ticker
    from matplotlib.gridspec import GridSpec
    
    # 根据时间范围截取数据
    start_idx = int(time_range[0] * sample_rate)
    end_idx = int(time_range[1] * sample_rate)
    data = data[start_idx:end_idx]

    # 重采样：降低采样率以加快处理速度
    if resample_factor > 1:
        data = decimate(data, resample_factor)
        sample_rate = sample_rate / resample_factor

    # 去直流分量：减去均值，消除低频偏移
    if remove_dc:
        data = data - np.mean(data)

    # 去趋势：去除线性趋势，避免影响频谱分析
    if detrend:
        data = scipy_detrend(data)

    # 加窗校正因子：补偿窗函数造成的能量损失
    window_correction = {
        "无": 1.0,
        "汉宁窗": 2.0,           # 1/0.5
        "汉明窗": 1.0 / 0.54,    # ≈1.85
        "布莱克曼窗": 1.0 / 0.42  # ≈2.38
    }
    correction_factor = window_correction.get(window_type, 1.0)

    # 计算瀑布图数据：使用短时傅里叶变换(STFT)
    window_size = int(sample_rate * 0.5)  # 窗口大小：0.5秒
    analysis_interval = int(sample_rate * 0.2)  # 分析间隔：0.2秒
    num_windows = (len(data) - window_size) // analysis_interval + 1

    waterfall_data = []
    times = []

    for i in range(num_windows):
        start = i * analysis_interval
        end = start + window_size
        window_data = data[start:end].copy()  # 使用已预处理的data，不再重复去直流/去趋势
        
        # 仅对每个窗口加窗（用于STFT）
        if window_type != "无":
            if window_type == "汉宁窗":
                w = np.hanning(len(window_data))
            elif window_type == "汉明窗":
                w = np.hamming(len(window_data))
            elif window_type == "布莱克曼窗":
                w = np.blackman(len(window_data))
            window_data = window_data * w
        
        # 计算每个窗口的FFT，并乘以校正因子补偿窗函数能量损失
        window_fft = np.fft.fft(window_data)
        window_mag = np.abs(window_fft)[:window_size // 2] * 2 / window_size * correction_factor
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

    # 计算主频：在选取的频率范围和时间窗口内，排除指定频率以下的低频干扰
    min_main_freq = max(min_freq_for_main, freq_range[0]) if freq_range is not None else min_freq_for_main
    valid_mask = waterfall_freq >= min_main_freq
    valid_spectrum = cross_spectrum.copy()
    valid_spectrum[~valid_mask] = 0
    main_freq_idx = np.argmax(valid_spectrum)
    main_freq = waterfall_freq[main_freq_idx]
    main_amp = valid_spectrum[main_freq_idx]

    # 左侧曲线图：带通滤波后的时域信号
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
    # 频域图横坐标从用户设置的freq_range[0]开始显示
    ax_top.set_xlim(freq_range[0], freq_range[1])
    # 计算纵坐标自适应范围：仅参考min_freq_for_main之后的数据，避免低频噪声影响
    plot_freq_start = max(min_freq_for_main, freq_range[0]) if freq_range is not None else min_freq_for_main
    visible_mask = (waterfall_freq >= plot_freq_start) & (waterfall_freq <= freq_range[1])
    visible_spectrum = cross_spectrum[visible_mask]
    spec_min = np.min(visible_spectrum)
    spec_max = np.max(visible_spectrum)
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
    if colorbar_max is not None:
        log_max = np.log10(colorbar_max)
    else:
        log_max = np.max(waterfall_log)
    if colorbar_min is not None:
        log_min = np.log10(colorbar_min)
    else:
        log_min = np.min(waterfall_log)
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

    # 色条刻度设置：增加刻度数量，确保最大值和最小值都有刻度
    # 生成包含最小值和最大值的等间距刻度
    num_ticks = 15  # 刻度数量
    tick_positions = np.linspace(log_min, log_max, num_ticks)
    cbar.set_ticks(tick_positions)
    cbar.formatter = ticker.FuncFormatter(log_to_real)
    cbar.update_ticks()

    return fig