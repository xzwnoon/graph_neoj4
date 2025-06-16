"""
模块: config.settings
描述: 提供项目的所有配置信息。
"""
from typing import Dict, Any

def get_settings() -> Dict[str, Any]:
    """
    提供项目的所有配置信息。
    在实际应用中，这些配置应从外部文件（如 config.toml, .env）安全加载，
    而不是硬编码在代码中。为了演示目的，此处进行硬编码。

    Returns:
        Dict[str, Any]: 包含所有配置的字典。
    """
    return {
        "llm": {
            "api_key": "sk-0bb6e4dae7ed44b58a8585b363c95dfb",
            "api_base" : "https://api.deepseek.com/v1", 
            "max_tokens" : 8192,
            "temperature" : 0.5,
            "model": "deepseek-chat",
            "timeout": 120,  # 请求超时时间（秒）
            "max_concurrency": 8,  # 最大并发线程数，提升抽取速度
        },
        "neo4j": {
            "uri": "bolt://localhost:7687",  # Neo4j 数据库URI
            "user": "neo4j",                 # 用户名
            "password": "GHr17wA66RyrS-VICFGCifLQVswtUydL0PCXKTx25Z4",     # !!!重要: 请替换为您的Neo4j密码
        },
        "text_chunking": {
            "max_chunk_size": 1000, # 文本分块的最大大小 (characters)
            "overlap_size": 50,    # 块之间的重叠大小 (characters)
        },
        "logging": {
            "level": "INFO", # 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            "format": "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
        },
        "confidence_rules": {
            "main_spo": 1.0,                # LLM主干SPO置信度
            "overlap": 0.8,                 # 规则法与主干重叠置信度
            "high_freq_base": 0.5,          # 高频规则法基础分
            "high_freq_step": 0.1,          # 高频规则法每多出现一次增加
            "high_freq_max": 0.8,           # 高频规则法最大分
            "low_freq": 0.3                  # 低频规则法置信度
        }
    }