import sys
import logging
from pathlib import Path
from loguru import logger as _logger

class LogManager:
    """
    Loguru 封装类：支持按名字区分模块，支持多文件，支持标准库接管
    """
    
    # 用于记录已经配置过独立文件的模块名称，防止重复 add handler
    _configured_modules = set()

    # 默认的日志格式
    _log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>[{extra[module_name]}]</cyan> "
        "<cyan>{name}:{line}</cyan> - "
        "<level>{message}</level>"
    )

    @classmethod
    def setup(cls, log_dir="logs", global_level="INFO", retention="10 days"):
        """
        全局初始化：设置控制台输出和默认的全局日志文件
        只需在程序入口调用一次
        """
        # 1. 移除 Loguru 默认的 handler，避免重复
        _logger.remove()
        
        # 2. 确保日志文件夹存在
        log_path = Path(log_dir)
        if not log_path.exists():
            log_path.mkdir(parents=True)

        # 3. 控制台输出 (输出所有模块日志)
        _logger.add(
            sys.stderr,
            format=cls._log_format,
            level=global_level,
            colorize=True
        )

        # 4. 全局日志文件 (汇总所有模块日志到 app_all.log)
        _logger.add(
            log_path / "app_all.log",
            rotation="10 MB",      # 文件过大自动切分
            retention=retention,   # 最长保留时间
            compression="zip",     # 自动压缩旧文件
            encoding="utf-8",
            enqueue=True,          # 异步写入，线程安全
            format=cls._log_format,
            level=global_level
        )

        # 5. 拦截标准 logging 库的日志 (让 requests, urllib3 等第三方库日志也走这里)
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    @classmethod
    def get_logger(cls, name: str, filename: str = None, level: str = "INFO", rotation="10 MB", retention="10 days", filter_func=None):
        """
        获取一个带名字的 Logger，并支持为该 Logger 设置独立的文件输出。

        Args:
            name (str): 模块名称 (例如 'Database', 'Network').
            filename (str, optional): 如果提供，该模块的日志会**额外**输出到这个文件.
            level (str, optional): 独立日志文件的等级 (默认 "INFO").
            rotation (str, optional): 独立日志文件的轮转规则 (默认 "10 MB").
            retention (str, optional): 独立日志文件的保留规则 (默认 "10 days").
            filter_func (callable, optional): 自定义过滤函数. 
                                              如果不传且提供了 filename，默认只记录当前模块(name)的日志.

        Returns:
            logger: 绑定了 module_name 的 logger 实例.

        Usage Examples (用法示例):

            # 1. 【基础用法】仅获取 Logger，日志去往控制台和全局 app_all.log
            >>> log = get_logger("System")
            >>> log.info("系统启动")

            # 2. 【进阶用法】将 Database 模块的日志单独存入 logs/db.log
            #    注意：该配置全局生效，即使多次调用 get_logger 也只会添加一次文件 Handler
            >>> db_log = get_logger("Database", filename="logs/db.log")
            >>> db_log.info("这条日志会同时出现在 app_all.log 和 db.log")

            # 3. 【高级用法】只把 Network 模块的 ERROR 级别日志存入 network_error.log
            >>> def only_error(record):
            ...     return record["extra"].get("module_name") == "Network" and record["level"].name == "ERROR"
            >>> net_log = get_logger("Network", filename="logs/network_error.log", filter_func=only_error)
            >>> net_log.info("这条不会进 error log")
            >>> net_log.error("这条会进 error log")
        """
        # 1. 绑定模块名，返回一个新的 logger 上下文
        new_logger = _logger.bind(module_name=name)

        # 2. 如果指定了 filename，且该模块尚未配置过独立文件，则添加 handler
        #    利用 _configured_modules 集合防止重复添加 handler
        if filename and name not in cls._configured_modules:
            
            # 默认过滤器：确保这个文件只包含当前模块的日志
            # 如果用户没有提供 filter_func，我们就用这个默认逻辑
            if filter_func is None:
                def default_filter(record):
                    return record["extra"].get("module_name") == name
                target_filter = default_filter
            else:
                target_filter = filter_func

            # 添加文件 Handler
            _logger.add(
                filename,
                filter=target_filter, # 关键：决定哪些日志能进这个文件
                level=level,
                rotation=rotation,
                retention=retention,
                compression="zip",
                encoding="utf-8",
                enqueue=True,
                format=cls._log_format
            )
            
            # 标记为已配置
            cls._configured_modules.add(name)

        return new_logger

class InterceptHandler(logging.Handler):
    """
    将标准 logging 库的日志重定向到 Loguru
    """
    def emit(self, record):
        try:
            level = _logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        # 将标准库的 name (如 urllib3) 绑定为 module_name
        _logger.bind(module_name=record.name).opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

# --- 初始化 ---
# 默认执行一次全局 Setup，保证导入即用
LogManager.setup()

# 暴露给外部使用的主要函数
get_logger = LogManager.get_logger

# ==========================================
#               使用演示区
# ==========================================
if __name__ == "__main__":
    # 这是一个演示，你可以直接运行这个文件查看效果
    
    print("--- 开始日志测试 ---")

    # 1. 基础用法
    sys_log = get_logger("System")
    sys_log.info("系统初始化完成 (控制台 + app_all.log)")

    # 2. 进阶用法：独立的数据库日志
    # 所有的 Database 日志都会额外写入 logs/db_ops.log
    db_log = get_logger("Database", filename="logs/db_ops.log")
    db_log.info("连接数据库成功 (控制台 + app_all.log + db_ops.log)")
    db_log.warning("查询耗时过长")

    # 3. 高级用法：只记录错误的 Network 日志
    def error_only(record):
        return record["extra"]["module_name"] == "Network" and record["level"].name == "ERROR"

    net_log = get_logger("Network", filename="logs/network_critical.log", filter_func=error_only)
    
    net_log.info("网络心跳正常 (只会进 app_all.log)")
    net_log.error("网络连接断开！ (会进 app_all.log + network_critical.log)")

    # 4. 测试异常捕获 (带变量值)
    try:
        x = 10
        y = 0
        z = x / y
    except ZeroDivisionError:
        sys_log.exception("捕获到一个除零异常")

    print("--- 日志测试结束，请查看 logs/ 文件夹 ---")