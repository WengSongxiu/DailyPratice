# -*- coding: utf-8 -*-
"""
定义文件目录
"""
import os

project_name = 'ml-log'  # 请根据实际改成你的项目名
current_dir = os.path.dirname(__file__)
project_path = os.path.dirname(current_dir)
log_data_dir = os.path.join(project_path, "log")  # 存放日志的文件夹
log_file_path = os.path.join(log_data_dir, '{}.log'.format(project_name))  # 日志的文件名

