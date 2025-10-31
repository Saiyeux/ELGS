#!/usr/bin/env python3
"""
ELGS Qt GUI Application
主入口文件
"""
import os
import sys

# 修复Qt平台插件冲突问题
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

from PyQt5.QtWidgets import QApplication
from EDGS.core.main_window import ELGSMainWindow

def main():
    app = QApplication(sys.argv)
    window = ELGSMainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()