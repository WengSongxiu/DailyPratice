# -*- coding: utf-8 -*-
"""
日志打印控制
"""

import logging.config
import os
import sys
from logging import LogRecord

from conf import file_conf

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

# 研发时的模式是debug，部署的时候改为info,warning或者error
LOG_LEVEL = logging.DEBUG


class ColorFormatter(logging.Formatter):
    """
    通常用于Linux系统下，使控制台输出的日志带颜色

    """
    log_colors = {
        'CRITICAL': '\033[0;31m',
        'ERROR': '\033[0;33m',
        'WARNING': '\033[0;35m',
        'INFO': '\033[0;32m',
        'DEBUG': '\033[0;00m',
    }

    def format(self, record: LogRecord):
        s = super().format(record)
        level_name = record.levelname
        if level_name in self.log_colors:
            return self.log_colors[level_name] + s + '\033[0m'
        return s


LOGGER = {
    'version': 1,
    'disable_existing_loggers': True,  # 为了兼容未来的版本
    'formatters': {
        'console': {
            'class': 'log.ColorFormatter',
            'format': '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
        },
        file_conf.project_name: {
            'class': 'logging.Formatter',
            'format': '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'level': LOG_LEVEL,
            'class': 'logging.StreamHandler',
            'formatter': 'console',
        },
        file_conf.project_name: {
            'level': LOG_LEVEL,
            'class': 'logging.handlers.RotatingFileHandler',
            'backupCount': 3,  # 日志最多有3个，超过3个则会覆盖以前的
            'maxBytes': 10 * 1024 * 1024,  # 每个log文件最大是10M
            'formatter': file_conf.project_name,
            'filename': file_conf.log_file_path,
            'encoding': 'utf-8',
        },
    },
    'loggers': {
        file_conf.project_name: {
            'handlers': [file_conf.project_name, 'console'],
            'level': LOG_LEVEL,
        },
    }
}

logging.config.dictConfig(LOGGER)
logger = logging.getLogger(file_conf.project_name)
