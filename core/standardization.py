"""
模块: core.standardization
描述: 标准化知识图谱中的实体名称。
"""
import requests
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# 定义实体标准化/解析的Prompt模板
STANDARDIZATION_SYSTEM_PROMPT = """
你是一个实体解析专家。你的任务是分析一个实体名称列表，识别指向同一真实世界概念的变体，并为每一组变体提供一个统一的、最具代表性的“标准名称”。

规则：
1.  **分组**: 将指代同一实体的所有名称（包括缩写、别名、不同拼写）归为一组。
2.  **选择标准名**: 从每组中选择一个最常用、最正式或最完整的名称作为标准名称。
3.  **JSON格式**: 结果必须是一个严格的JSON对象。对象的键是“标准名称”，值是包含所有变体（包括标准名称自身）的数组。
4.  **只包含需标准化的**: 如果一个实体没有变体，不要将其包含在输出中。

示例输入: ["The United States", "U.S.", "America", "United States of America", "steam engine"]
示例输出格式:
{
  "united states of america": ["the united states", "u.s.", "america", "united states of america"]
}
"""

def get_standardization_user_prompt(entities: List[str]) -> str:
    """生成用于实体标准化的User Prompt"""
    return f"请根据以上规则，对以下实体列表进行标准化处理。列表如下:\n\n{json.dumps(entities, indent=2)}"


def _call_llm_api(system_prompt: str, user_prompt: str, config: Dict[str, Any]) -> Optional[Dict[str, List[str]]]:
    """内部辅助函数，用于调用LLM API并获取标准化映射。"""
    api_config = config.get("llm", {})
    api_key, api_base, model, timeout = (
        api_config.get("api_key"),
        api_config.get("api_base"),
        api_config.get("model"),
        api_config.get("timeout", 120),
    )

    if not all([api_key, api_base, model]):
        logger.error("LLM配置不完整 (api_key, api_base, model 必须提供)。")
        return None

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    try:
        response = requests.post(f"{api_base}/chat/completions", headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        content_str = response.json()['choices'][0]['message']['content']
        try:
            return json.loads(content_str)
        except json.JSONDecodeError as e:
            logger.error(f"标准化LLM返回内容JSON解析失败，原始内容片段: {content_str[:500]}...", exc_info=True)
            # 可选：尝试修正常见格式问题后再次解析
            return None
    except Exception as e:
        logger.error(f"调用或解析实体标准化LLM API时出错: {e}", exc_info=True)
        return None


def standardize_entities(triplets: List[Dict[str, str]], config: Dict[str, Any]) -> Dict[str, str]:
    """
    对SPO三元组列表中的所有实体进行标准化。

    Args:
        triplets (List[Dict[str, str]]): 从文本中抽取的SPO三元组列表。
        config (Dict[str, Any]): 项目配置。

    Returns:
        Dict[str, str]: 一个映射字典，键是原始实体名，值是其对应的标准实体名。
    """
    # 提取所有独特的主语和宾语实体
    entities = set()
    for t in triplets:
        entities.add(t['subject'])
        entities.add(t['object'])
    
    unique_entities = sorted(list(entities))
    logger.info(f"开始对 {len(unique_entities)} 个独立实体进行标准化...")

    user_prompt = get_standardization_user_prompt(unique_entities)
    standardization_groups = _call_llm_api(STANDARDIZATION_SYSTEM_PROMPT, user_prompt, config)

    if not standardization_groups:
        logger.warning("未能从LLM获取到实体标准化分组，将使用原始实体名。")
        return {entity: entity for entity in unique_entities}

    # 创建从变体到标准名称的映射
    variant_to_standard_map = {}
    for standard_name, variants in standardization_groups.items():
        for variant in variants:
            variant_to_standard_map[variant] = standard_name
    
    # 为所有实体创建最终映射，未被标准化的实体映射到自身
    final_map = {}
    for entity in unique_entities:
        final_map[entity] = variant_to_standard_map.get(entity, entity)

    num_standardized = len([k for k, v in final_map.items() if k != v])
    logger.info(f"实体标准化完成。{num_standardized} 个实体被映射到标准名称。")
    return final_map