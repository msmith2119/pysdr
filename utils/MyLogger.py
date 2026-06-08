import json
from enum import IntEnum


class LogLevel(IntEnum):
    INFO = 1
    WARN = 2
    ERROR = 3


class MyLogger:
    # Class-level (static) variable for current log level
    _current_level = LogLevel.INFO

    @classmethod
    def set_level(cls, level: LogLevel):
        cls._current_level = level

    @classmethod
    def log(cls, message: str, level: LogLevel):
        if level >= cls._current_level:
            print(f"[{level.name}] {message}")
    @classmethod
    def log_dict(cls,d,level: LogLevel):
        if level >= cls._current_level:
            cls.log(json.dumps(d, indent=2), LogLevel.INFO)


    @classmethod
    def error(cls,msg):
        cls.log(msg,LogLevel.ERROR)