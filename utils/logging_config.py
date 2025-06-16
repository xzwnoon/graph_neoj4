"""
模块: utils.logging_config
描述: 设置全局日志记录器。
"""
import logging
from typing import Dict, Any

def setup_logging(config: Dict[str, Any]) -> None:
    """
    根据提供的配置设置全局日志记录器。

    Args:
        config (Dict[str, Any]): 包含日志配置的字典 ('level', 'format').
    """
    log_level = config.get("level", "INFO").upper()
    log_format = config.get("format", "%(asctime)s - [%(levelname)s] - %(message)s")

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler()  # 输出到控制台
        ]
    )
    # 减少第三方库的日志噪音
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成，级别: {log_level}")