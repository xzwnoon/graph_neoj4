"""
模块: core.chunking
描述: 将长文本分割成小块。
"""
import logging
from typing import List, Dict, Any
import tiktoken  # 需提前安装tiktoken

logger = logging.getLogger(__name__)

def chunk_text(text: str, config: Dict[str, Any]) -> List[str]:
    """
    将长文本分割成较小的、有重叠的块。
    这是一个简单的基于字符数的实现。

    Args:
        text (str): 待分割的原始文本。
        config (Dict[str, Any]): 包含分块参数的配置字典 ('max_chunk_size', 'overlap_size')。

    Returns:
        List[str]: 分割后的文本块列表。
    """
    max_size = config.get("max_chunk_size", 2000)
    overlap = config.get("overlap_size", 200)
    
    if not text:
        logger.warning("输入的文本为空，无法进行分块。")
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_size - overlap
    
    logger.info(f"文本成功分割成 {len(chunks)} 个块。")
    return chunks

def estimate_tokens(text: str, model: str = 'gpt-3.5-turbo') -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding('cl100k_base')
    return len(enc.encode(text))

def is_informative_chunk(chunk: str, min_words: int = 10) -> bool:
    # 过滤掉纯空白、纯符号、极短内容、疑似表格/作者列表等
    text = chunk.strip()
    if not text:
        return False
    # 过滤掉大部分为非字母数字的内容
    alpha_ratio = sum(c.isalnum() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.2:
        return False
    # 过滤掉词数极少的内容
    if len(text.split()) < min_words:
        return False
    return True

def smart_chunk_text(text: str, config: Dict[str, Any]) -> List[str]:
    """
    根据LLM最大token数和Prompt长度动态分块，支持chunk_tokens配置，过滤无效块。
    """
    llm_conf = config.get('llm', {})
    chunk_tokens = llm_conf.get('chunk_tokens', 2048)
    max_tokens = llm_conf.get('max_tokens', 4096)
    model = llm_conf.get('model', 'gpt-3.5-turbo')
    prompt_tokens = estimate_tokens('SPO抽取Prompt', model) + 100  # 预留
    chunk_tokens = min(chunk_tokens, max_tokens - prompt_tokens)
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding('cl100k_base')
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk = enc.decode(tokens[start:end])
        if is_informative_chunk(chunk):
            chunks.append(chunk)
        start += chunk_tokens
    logger.info(f"智能分块后共 {len(chunks)} 块，每块约 {chunk_tokens} tokens")
    return chunks