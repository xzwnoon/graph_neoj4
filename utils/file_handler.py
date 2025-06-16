"""
模块: utils.file_handler
描述: 读取文本文件内容。
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def read_text_file(file_path: str) -> Optional[str]:
    """
    读取指定路径的文本文件内容。

    Args:
        file_path (str): 文本文件的路径。

    Returns:
        Optional[str]: 文件内容字符串，如果读取失败则返回 None。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"成功读取文件: {file_path}")
        return content
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        return None
    except Exception as e:
        logger.error(f"读取文件时发生未知错误 {file_path}: {e}", exc_info=True)
        return None