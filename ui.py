"""
UI模块 - 包含主窗口界面和交互逻辑
"""
import os
import sys
import subprocess
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QFileDialog, QGroupBox,
                             QGridLayout, QMessageBox, QAction, QMenuBar, QListWidget,
                             QListWidgetItem, QCheckBox, QComboBox, QFrame, QProgressDialog,
                             QInputDialog, QDialog, QTextEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from config import DEFAULT_PARAMS
from signal_processor import read_sts, generate_plot


class MainWindow(QMainWindow):
    """主窗口类：实现GUI界面和交互逻辑"""
    def __init__(self):
        super().__init__()
        from __init__ import __version__
        self.setWindowTitle(f"振动加速度数据分析工具 v{__version__}")
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
        self.colorbar_min = None    # 色条最小值，None表示自适应
        self.window_type = DEFAULT_PARAMS['window_type']  # 加窗类型
        self.min_freq_for_main = DEFAULT_PARAMS['min_freq_for_main']  # 主频搜索最小频率

        # 创建菜单栏
        self._setup_menu()
        
        # 创建主界面布局
        self._setup_ui()

    def _setup_menu(self):
        """设置菜单栏"""
        menubar = self.menuBar()
        
        # 设置菜单
        settings_menu = menubar.addMenu("设置")
        
        clear_folder_action = QAction("清除文件夹选择", self)
        clear_folder_action.triggered.connect(self.clear_folder)
        settings_menu.addAction(clear_folder_action)
        
        self.colorbar_action = QAction("修改瀑布图色条上限（默认）", self)
        self.colorbar_action.triggered.connect(self.set_colorbar_max)
        settings_menu.addAction(self.colorbar_action)
        
        self.colorbar_min_action = QAction("修改瀑布图色条最小值（默认）", self)
        self.colorbar_min_action.triggered.connect(self.set_colorbar_min)
        settings_menu.addAction(self.colorbar_min_action)
        
        self.min_freq_action = QAction(f"主频搜索起始频率（{self.min_freq_for_main}Hz）", self)
        self.min_freq_action.triggered.connect(self.set_min_freq_for_main)
        settings_menu.addAction(self.min_freq_action)
        
        # 帮助菜单
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

    def _setup_ui(self):
        """设置主界面布局"""
        # 创建主布局 - 三列布局
        main_layout = QHBoxLayout()

        # 左侧控制面板
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)

        # 右侧控制面板 - 文件选择和文件列表
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel)

        # 右侧图像显示区
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout()
        self.plot_widget.setLayout(self.plot_layout)
        main_layout.addWidget(self.plot_widget)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _create_control_panel(self):
        """创建左侧控制面板"""
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

        # 创建参数输入框
        self._create_param_input(param_layout, label_font, edit_font)

        param_group.setLayout(param_layout)
        control_layout.addWidget(param_group)

        # 前处理设置
        preprocess_group = self._create_preprocess_group(edit_font, label_font)
        control_layout.addWidget(preprocess_group)

        # 按钮
        self._create_buttons(control_layout)

        control_layout.addStretch()
        return control_panel

    def _create_param_input(self, layout, label_font, edit_font):
        """创建参数输入框"""
        params = [
            ("时间下限 (s):", "time_min", str(DEFAULT_PARAMS['time_min'])),
            ("时间上限 (s):", "time_max", str(DEFAULT_PARAMS['time_max'])),
            ("频率下限 (Hz):", "freq_min", str(DEFAULT_PARAMS['freq_min'])),
            ("频率上限 (Hz):", "freq_max", str(DEFAULT_PARAMS['freq_max'])),
            ("采样频率 (Hz):", "sample_rate", str(DEFAULT_PARAMS['sample_rate'])),
            ("频谱截取时刻 (s):", "cross_time", str(DEFAULT_PARAMS['cross_time'])),
            ("时间范围 (s):", "cross_time_window", str(DEFAULT_PARAMS['cross_time_window'])),
            ("滤波中心频率 (Hz):", "filter_freq", str(DEFAULT_PARAMS['filter_freq'])),
            ("频率范围 (Hz):", "filter_freq_window", str(DEFAULT_PARAMS['filter_freq_window'])),
        ]

        for label_text, attr_name, default_value in params:
            v_layout = QVBoxLayout()
            label = QLabel(label_text)
            label.setFont(label_font)
            v_layout.addWidget(label)
            
            edit = QLineEdit(default_value)
            edit.setFont(edit_font)
            setattr(self, attr_name, edit)
            v_layout.addWidget(edit)
            layout.addLayout(v_layout)

    def _create_preprocess_group(self, edit_font, label_font):
        """创建前处理设置组"""
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
        self.resample_factor = QLineEdit(str(DEFAULT_PARAMS['resample_factor']))
        self.resample_factor.setFont(edit_font)
        resample_layout.addWidget(self.resample_factor)
        preprocess_layout.addLayout(resample_layout)

        # 窗函数选择
        window_layout = QVBoxLayout()
        window_label = QLabel("窗函数：")
        window_label.setFont(label_font)
        window_layout.addWidget(window_label)
        self.window_combo = QComboBox()
        self.window_combo.setFont(edit_font)
        self.window_combo.addItems(["无", "汉宁窗", "汉明窗", "布莱克曼窗"])
        self.window_combo.setCurrentText(DEFAULT_PARAMS['window_type'])
        self.window_combo.currentTextChanged.connect(self.on_window_changed)
        window_layout.addWidget(self.window_combo)
        preprocess_layout.addLayout(window_layout)

        preprocess_group.setLayout(preprocess_layout)
        return preprocess_group

    def _create_buttons(self, layout):
        """创建操作按钮"""
        btn_font = QFont()
        btn_font.setPointSize(10)
        
        self.generate_btn = QPushButton("生成图像")
        self.generate_btn.setFont(btn_font)
        self.generate_btn.setFixedHeight(35)
        self.generate_btn.clicked.connect(self.generate_image)
        layout.addWidget(self.generate_btn)

        self.save_btn = QPushButton("保存图像")
        self.save_btn.setFont(btn_font)
        self.save_btn.setFixedHeight(35)
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)

        self.save_all_btn = QPushButton("保存所有图像")
        self.save_all_btn.setFont(btn_font)
        self.save_all_btn.setFixedHeight(35)
        self.save_all_btn.clicked.connect(self.save_all_images)
        self.save_all_btn.setEnabled(False)
        layout.addWidget(self.save_all_btn)

    def _create_right_panel(self):
        """创建右侧文件选择面板"""
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        right_panel.setFixedWidth(380)

        # 文件选择
        file_group = self._create_file_selection_group()
        right_layout.addWidget(file_group)

        # 文件列表
        file_list_group = self._create_file_list_group()
        right_layout.addWidget(file_list_group)

        right_layout.addStretch()
        return right_panel

    def _create_file_selection_group(self):
        """创建文件选择组"""
        file_group = QGroupBox("文件选择")
        file_group.setFont(self.font)
        file_layout = QVBoxLayout()
        
        file_label_font = QFont()
        file_label_font.setPointSize(10)
        
        # 单文件选择
        self.file_label = QLabel("未选择文件")
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
        return file_group

    def _create_file_list_group(self):
        """创建文件列表组"""
        file_list_group = QGroupBox("STS文件列表")
        file_list_group.setFont(self.font)
        file_list_layout = QVBoxLayout()
        self.file_list_widget = QListWidget()
        self.file_list_widget.setFont(QFont("Consolas", 10))
        self.file_list_widget.itemClicked.connect(self.on_file_clicked)
        file_list_layout.addWidget(self.file_list_widget)
        file_list_group.setLayout(file_list_layout)
        return file_list_group

    def _adjust_time_range(self):
        """根据数据长度自动调整时间范围，确保不超出数据实际长度"""
        if self.data is None:
            return
        
        # 计算数据的实际时间长度（秒）
        max_time = len(self.data) / int(self.sample_rate.text())
        
        # 获取当前输入的时间范围
        try:
            time_min_val = float(self.time_min.text())
            time_max_val = float(self.time_max.text())
        except ValueError:
            return
        
        # 调整最小时间
        if time_min_val < 0:
            time_min_val = 0
        elif time_min_val > max_time:
            time_min_val = max_time
        
        # 调整最大时间
        if time_max_val > max_time:
            time_max_val = max_time
        elif time_max_val < time_min_val:
            time_max_val = time_min_val
        
        # 更新输入框
        self.time_min.setText(f"{time_min_val:.2f}")
        self.time_max.setText(f"{time_max_val:.2f}")

    def select_file(self):
        """选择单个STS文件并读取数据"""
        path, _ = QFileDialog.getOpenFileName(self, "选择STS文件", "", "STS文件 (*.sts)")
        if path:
            self.sts_path = path
            self.file_label.setText(os.path.basename(path))
            try:
                self.data = read_sts(path)
                self._adjust_time_range()
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
            
            for f in os.listdir(path):
                if f.lower().endswith('.sts'):
                    suffix = f.split('_')[-1] if '_' in f else f
                    num = 0
                    if '#' in f:
                        try:
                            num_str = f.split('#')[1].split('_')[0]
                            num = int(num_str)
                        except:
                            num = 0
                    file_pairs.append((num, suffix, f))
            
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
        self.colorbar_min = None
        self.colorbar_action.setText("修改瀑布图色条上限（默认）")
        self.colorbar_min_action.setText("修改瀑布图色条最小值（默认）")
        self.window_type = DEFAULT_PARAMS['window_type']
        self.window_combo.setCurrentText(DEFAULT_PARAMS['window_type'])

    def set_min_freq_for_main(self):
        """设置主频搜索起始频率，并自动更新图像"""
        text, ok = QInputDialog.getText(self, "设置主频搜索起始频率", 
                                         "请输入最小频率（Hz，用于排除低频噪声）：",
                                         text=str(self.min_freq_for_main))
        if ok:
            try:
                val = float(text)
                if val < 0:
                    QMessageBox.warning(self, "警告", "频率不能为负数！")
                    return
                self.min_freq_for_main = val
                self.min_freq_action.setText(f"主频搜索起始频率（{val:g}Hz）")
                self._update_plot()
            except ValueError:
                QMessageBox.warning(self, "警告", "请输入有效的数字！")

    def on_window_changed(self, text):
        """窗函数选择改变时记录选择，不自动更新图像"""
        self.window_type = text

    def set_colorbar_max(self):
        """设置色条最大值"""
        default_val = "" if self.colorbar_max is None else str(self.colorbar_max)
        text, ok = QInputDialog.getText(self, "设置色条最大值", "请输入色条最大值（留空表示自适应）：", text=default_val)
        if ok:
            if text.strip() == "":
                self.colorbar_max = None
                self.colorbar_action.setText("修改瀑布图色条上限（默认）")
            else:
                try:
                    val = float(text)
                    self.colorbar_max = val
                    self.colorbar_action.setText(f"修改瀑布图色条上限（{val:g}）")
                except ValueError:
                    QMessageBox.warning(self, "警告", "请输入有效的数字！")
                    return
            self._update_plot()

    def set_colorbar_min(self):
        """设置色条最小值"""
        default_val = "" if self.colorbar_min is None else str(self.colorbar_min)
        text, ok = QInputDialog.getText(self, "设置色条最小值", "请输入色条最小值（留空表示自适应）：", text=default_val)
        if ok:
            if text.strip() == "":
                self.colorbar_min = None
                self.colorbar_min_action.setText("修改瀑布图色条最小值（默认）")
            else:
                try:
                    val = float(text)
                    self.colorbar_min = val
                    self.colorbar_min_action.setText(f"修改瀑布图色条最小值（{val:g}）")
                except ValueError:
                    QMessageBox.warning(self, "警告", "请输入有效的数字！")
                    return
            self._update_plot()

    def _update_plot(self):
        """更新当前显示的图像"""
        if not self.sts_path or self.data is None:
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

            fig = generate_plot(self.data, sample_rate, time_range, freq_range,
                               cross_time, cross_time_window,
                               filter_freq, filter_freq_window,
                               filename=os.path.basename(self.sts_path),
                               colorbar_max=self.colorbar_max,
                               colorbar_min=self.colorbar_min,
                               remove_dc=self.remove_dc.isChecked(),
                               detrend=self.detrend.isChecked(),
                               window_type=self.window_type,
                               resample_factor=resample_factor,
                               min_freq_for_main=self.min_freq_for_main)

            for i in reversed(range(self.plot_layout.count())):
                self.plot_layout.itemAt(i).widget().deleteLater()

            # 关闭旧的figure对象，防止内存泄漏
            if self.figure is not None:
                plt.close(self.figure)

            self.figure = fig
            self.canvas = FigureCanvas(self.figure)
            toolbar = NavigationToolbar(self.canvas, self)
            self.plot_layout.addWidget(toolbar)
            self.plot_layout.addWidget(self.canvas)
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新图像失败：{str(e)}")

    def on_file_clicked(self, item):
        """点击文件列表中的文件时显示对应图像"""
        idx = self.file_list_widget.row(item)
        if idx < len(self.file_list):
            file_path = self.file_list[idx]
            try:
                data = read_sts(file_path)
                self.sts_path = file_path
                self.data = data
                self._adjust_time_range()
                
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
                                   colorbar_min=self.colorbar_min,
                                   remove_dc=self.remove_dc.isChecked(),
                                   detrend=self.detrend.isChecked(),
                                   window_type=self.window_type,
                                   resample_factor=int(self.resample_factor.text()),
                                   min_freq_for_main=self.min_freq_for_main)

                for i in reversed(range(self.plot_layout.count())):
                    self.plot_layout.itemAt(i).widget().deleteLater()

                # 关闭旧的figure对象，防止内存泄漏
                if self.figure is not None:
                    plt.close(self.figure)

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
        """生成当前文件的分析图像"""
        if not self.sts_path:
            QMessageBox.warning(self, "警告", "请先选择STS文件！")
            return

        self._adjust_time_range()

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
                                       colorbar_min=self.colorbar_min,
                                       remove_dc=self.remove_dc.isChecked(),
                                       detrend=self.detrend.isChecked(),
                                       window_type=self.window_type,
                                       resample_factor=int(self.resample_factor.text()),
                                       min_freq_for_main=self.min_freq_for_main)

            for i in reversed(range(self.plot_layout.count())):
                self.plot_layout.itemAt(i).widget().deleteLater()

            # 关闭旧的figure对象，防止内存泄漏
            if self.figure is not None:
                plt.close(self.figure)

            self.canvas = FigureCanvas(self.figure)
            toolbar = NavigationToolbar(self.canvas, self)
            self.plot_layout.addWidget(toolbar)
            self.plot_layout.addWidget(self.canvas)
            self.canvas.draw()
            self.save_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"图像生成失败: {str(e)}")

    def save_image(self):
        """保存当前图像"""
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
        """批量保存文件夹中所有STS文件的图像"""
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

        output_folder = os.path.join(self.folder_path, "图像")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        success_count = 0
        fail_count = 0
        fail_files = []

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
                file_max_time = len(data) / sample_rate
                
                current_time_min = time_range[0]
                current_time_max = time_range[1]
                
                if current_time_min < 0:
                    current_time_min = 0
                elif current_time_min > file_max_time:
                    current_time_min = file_max_time
                
                if current_time_max > file_max_time:
                    current_time_max = file_max_time
                elif current_time_max < current_time_min:
                    current_time_max = current_time_min
                
                adjusted_time_range = (current_time_min, current_time_max)
                
                fig = generate_plot(data, sample_rate, adjusted_time_range, freq_range,
                                   cross_time, cross_time_window,
                                   filter_freq, filter_freq_window,
                                   filename=os.path.basename(file_path),
                                   colorbar_max=self.colorbar_max,
                                   colorbar_min=self.colorbar_min,
                                   remove_dc=self.remove_dc.isChecked(),
                                   detrend=self.detrend.isChecked(),
                                   window_type=self.window_type,
                                   resample_factor=resample_factor,
                                   min_freq_for_main=self.min_freq_for_main)

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
            folder_path = os.path.normpath(self.folder_path)
            subprocess.Popen(f'explorer "{folder_path}"')
        else:
            QMessageBox.critical(self, "错误", msg)

    def show_about(self):
        """显示关于对话框"""
        from __init__ import __version__
        dialog = QDialog(self)
        dialog.setWindowTitle("关于")
        dialog.resize(600, 500)
        
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml(self._get_about_html(__version__))
        layout.addWidget(text_edit)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def _get_about_html(self, version):
        """获取关于对话框HTML内容"""
        return (
            f"<h3>振动加速度数据分析工具 v{version}</h3>"
            "<p>作者: StupidMoonlight<br>"
            "GitHub: <a href='https://github.com/StupidMoonlight'>https://github.com/StupidMoonlight</a></p>"
            "<p>用于振动加速度信号的FFT频谱分析和可视化。</p>"
            "<h4>v1.2.1 更新内容：</h4>"
            "<ul>"
            "<li>修复设置菜单更新图像时缺少工具栏的问题</li>"
            "<li>代码重构：将单文件拆分为多个模块，提高可维护性</li>"
            "<li>新增时间范围自动调整功能：当输入的时间超出文件实际长度时，程序会自动调整为文件的最大时间</li>"
            "<li>修复频繁切换文件时可能导致的内存泄漏问题</li>"
            "</ul>"
            "<h4>v1.2 更新内容：</h4>"
            "<ul>"
            "<li>将窗函数选择从设置菜单移至前处理设置界面，操作更直观</li>"
            "<li>窗函数默认值改为汉宁窗</li>"
            "<li>优化STFT分析：分析间隔改为0.2秒，提高时间分辨率</li>"
            "<li>增加色条刻度数量（15个），确保最大值和最小值都有刻度标注</li>"
            "<li>修复色条最小值自适应逻辑，默认真正自适应数据范围</li>"
            "<li>将色条改为DASP同款色条样式</li>"
            "<li>新增时间范围自动调整功能：当输入的时间超出文件实际长度时，程序会自动调整为文件的最大时间</li>"
            "</ul>"
            "<h4>v1.1 更新内容：</h4>"
            "<ul>"
            "<li>修复STFT重复预处理导致幅值错误的问题</li>"
            "<li>添加加窗校正因子，确保加窗后幅值准确</li>"
            "<li>添加主频搜索起始频率设置，排除低频噪声干扰</li>"
            "<li>添加色条最小值设置，可自定义色条范围</li>"
            "<li>移除错误的单位转换（数据默认为加速度值，无需转换）</li>"
            "<li>优化频域图纵坐标自适应逻辑</li>"
            "</ul>"
        )

    def show_usage(self):
        """显示使用说明对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("使用说明")
        dialog.resize(700, 500)
        
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml(self._get_usage_html())
        layout.addWidget(text_edit)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def _get_usage_html(self):
        """获取使用说明HTML内容"""
        return (
            "<h3>【使用说明】</h3>"
            "<h4>1. 选择数据文件：</h4>"
            "<ul>"
            "<li>点击<b>\"选择STS文件\"</b>按钮，选择单个要分析的.sts振动数据文件。</li>"
            "<li>点击<b>\"选择文件夹\"</b>按钮，选择包含多个.sts文件的文件夹，程序会自动列出所有STS文件，点击文件即可预览对应图像。</li>"
            "</ul>"
            "<h4>2. 设置参数：</h4>"
            "<ul>"
            "<li><b>时间上下限</b>：选择要分析的时间段范围（单位：秒）。当输入的时间超出文件实际长度时，程序会自动调整为文件的最大时间</li>"
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
            "<li><b>窗函数</b>：选择窗函数类型（无/汉宁窗/汉明窗/布莱克曼窗，默认汉宁窗）</li>"
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
            "<li><b>修改瀑布图色条上限</b>：设置色条的最大值（默认自适应）</li>"
            "<li><b>修改瀑布图色条最小值</b>：设置色条的最小值（默认自适应）</li>"
            "<li><b>主频搜索起始频率</b>：设置主频计算的最低频率阈值，排除低频噪声干扰（默认10Hz）。纵坐标自适应范围会参考该值之后的数据，避免低频噪声导致主频信号被压缩</li>"
            "</ul>"
            "<h3>【图像说明】</h3>"
            "<ul>"
            "<li><b>上方</b>：指定时刻的频域剖面图（FFT频谱），红色虚线标注主频</li>"
            "<li><b>中间</b>：瀑布图（频率-时间-幅值），两条红色虚线标注剖面位置</li>"
            "<li><b>左侧</b>：带通滤波后的时域信号</li>"
            "<li><b>右侧</b>：色条（对数刻度）</li>"
            "</ul>"
        )

    def show_disclaimer(self):
        """显示免责声明对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("免责声明")
        dialog.resize(600, 400)
        
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml(self._get_disclaimer_html())
        layout.addWidget(text_edit)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def _get_disclaimer_html(self):
        """获取免责声明HTML内容"""
        return (
            "<h3>免责声明</h3>"
            "<p>1. 本软件由作者独立开发，仅供学习和研究使用，不提供任何形式的明示或暗示的保证。</p>"
            "<p>2. 用户使用本软件进行数据分析所产生的结果，由用户自行负责。作者不对分析结果的准确性、可靠性或适用性作任何保证。</p>"
            "<p>3. 在任何情况下，作者或版权持有人均不对因使用本软件而产生的任何直接的、间接的、偶然的、特殊的或后果性的损害承担责任。</p>"
            "<p>4. 本软件的使用者应遵守相关法律法规，不得将本软件用于非法目的。使用者违反法律法规所产生的后果由使用者自行承担。</p>"
            "<p>5. 本软件的分析结果仅供参考，不应作为工程决策的唯一依据。对于涉及安全的工程应用，请使用经过认证的专业分析工具。</p>"
        )