
import logging
import colorlog
import datetime
import os

# 自定义时间格式化器，输出RFC3339格式
# 这里是对控制台而言 可以输出颜色 下为文件记录不能输出颜色 此处是重写时间戳记录
class RFC3339Formatter(colorlog.ColoredFormatter):#这里是继承colorlog.ColoredFormatter
    def formatTime(self, record, datefmt=None):#这里是重写方法
        # 创建RFC3339格式的时间戳，例如: 2025-07-24T02:04:23.466388Z
        dt = datetime.datetime.fromtimestamp(record.created)
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

# 创建文件格式化器（不带颜色）
class PlainRFC3339Formatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created)#LogRecord的屬性 是一种浮点数 可以被转化成标准字符串
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

# 控制台日志格式（带颜色，显示完整路径）
CONSOLE_FORMAT = "%(asctime)s[%(log_color)s%(levelname)s%(reset)s][%(pathname)s:%(lineno)d] %(message)s"
#logging模块再生成日志时会创建一个LogRecord对象 包含这些类型
#这里%()是字典式读取 相当于从LogRecord对象内部调取asctime属性的值 并且转化为字符串 输出到%占位的地方
# 文件日志格式（不带颜色，显示完整路径）
FILE_FORMAT = "%(asctime)s[%(levelname)s][%(pathname)s:%(lineno)d] %(message)s"

# 创建控制台格式化器（带颜色）
console_formatter = RFC3339Formatter(#这里是对colorlog.ColoredFormatter(logging.Formatter)对象初始化 其存在如下属性
    CONSOLE_FORMAT,
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
     secondary_log_colors={
        'module_color': {
            'DEBUG': 'blue',
            'INFO': 'blue',
            'WARNING': 'blue',
            'ERROR': 'blue',
            'CRITICAL': 'blue'
        },
        'file_color': {
            'DEBUG': 'purple',
            'INFO': 'purple',
            'WARNING': 'purple',
            'ERROR': 'purple',
            'CRITICAL': 'purple'
        }
    },
    style="%",
)

# 文件日志格式（不带颜色）
file_formatter = PlainRFC3339Formatter(FILE_FORMAT)

# 获取根日志记录器
logger = logging.getLogger()#logger是logging库实例化的对象 这里是不是没有

# 创建一个控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)
# 接收 Logger 发来的 LogRecord 对象，
# 然后利用你配置的 Formatter（即 console_formatter）将这个对象转换成可读的文本，并将其打印到屏幕上。

# 创建文件处理器
# 确保logs目录存在
os.makedirs('logs', exist_ok=True)

# 生成带时间戳的日志文件名
log_filename = f"logs/python_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.log"
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setFormatter(file_formatter)

# 将处理器添加到日志记录器
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# 示例日志消息
# logger.debug("This is a debug message")
# logger.info("This is an info message")
# logger.warning("This is a warning message")
# logger.error("This is an error message")
# logger.critical("This is a critical message")

# python
def getLogger(level=logging.INFO):
    logger.setLevel(level)#这里设定logger记录的信息级别
    #logger.info(”text“)
    return logger
