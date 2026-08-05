"""
主入口文件 - 程序启动入口
"""
import sys
import os

# 配置PyInstaller打包后的资源路径
if getattr(sys, 'frozen', False):
    # 如果是打包后的exe运行，设置_MEIPASS为临时解压目录
    base_path = sys._MEIPASS
else:
    # 如果是源码运行，使用当前目录
    base_path = os.path.dirname(os.path.abspath(__file__))

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from ui import MainWindow


def main():
    """程序主入口"""
    # 设置Windows应用ID，使任务栏图标正常显示
    if sys.platform == 'win32':
        try:
            import ctypes
            app_id = "StupidMoonlight.VibrationAnalyzer.v1.2.2"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except:
            pass
    
    # 设置高DPI支持（必须在创建QApplication之前）
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, False)
    
    # 创建应用
    app = QApplication(sys.argv)
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()