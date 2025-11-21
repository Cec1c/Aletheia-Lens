# 一些通用修饰器
import time


def timer_decorator(func):
    """
    过程耗时输出
    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__}运行时间：{end_time - start_time}秒")
        return result
    return wrapper

