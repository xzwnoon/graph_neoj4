"""
模块: core.extraction
描述: 从文本中抽取知识三元组。
"""
import requests
import json
import logging
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# 定制铝合金领域Prompt，补充完整关系链条
EXTRACTION_SYSTEM_PROMPT = """
你是材料科学领域的知识图谱专家，专注于铝合金相关科研文献。请从下列文本中抽取所有与“成分-影响-性能”、“热处理工艺-影响-组织结构”、“成分-热处理工艺-性能-组织结构”等相关的主语-谓语-宾语（SPO）三元组，尤其关注成分、热处理工艺、性能、组织结构之间的完整链式关系。

规则：
1.  **最大化信息**: 识别文本中所有可能的关系，确保知识的全面性。
2.  **优先抽取**：
    - 成分-影响-性能
    - 热处理工艺-影响-组织结构
    - 成分-影响-组织结构
    - 组织结构-影响-性能
    - 成分-热处理工艺-性能-组织结构（链式关系可拆分为多条SPO）
3.  **谓词简洁**: 谓词（Predicate）必须非常简洁，通常是1-3个单词的动词或动词短语。
4.  **实体规范**: 主语（Subject）和宾语（Object）应该是具体的实体名称。
5.  **小写输出**: 所有SPO的文本都应为小写，包括专有名词。
6.  **JSON格式**: 结果必须是一个严格的JSON数组，每个元素是一个包含 "subject", "predicate", "object" 键的对象。不要在JSON之外添加任何解释性文字。
7.  **只抽取最核心的SPO关系**，如内容过长可分段抽取。
"""

# 分步Prompt
COMPONENT_PROMPT = "你是材料科学专家，请从下列文本中抽取所有铝合金相关的成分（如元素、含量、配比等），输出为JSON数组，每个元素为一个成分字符串。不要输出解释性文字。"
PROCESS_PROMPT = "你是材料科学专家，请从下列文本中抽取所有铝合金相关的热处理工艺（如淬火、时效、退火等），输出为JSON数组，每个元素为一个工艺字符串。不要输出解释性文字。"
PERFORMANCE_PROMPT = "你是材料科学专家，请从下列文本中抽取所有铝合金相关的性能指标（如强度、延伸率、硬度等），输出为JSON数组，每个元素为一个性能字符串。不要输出解释性文字。"
STRUCTURE_PROMPT = "你是材料科学专家，请从下列文本中抽取所有铝合金相关的组织结构（如晶粒、析出相等），输出为JSON数组，每个元素为一个组织结构字符串。不要输出解释性文字。"

def get_extraction_user_prompt(text_chunk: str) -> str:
    """生成用于SPO抽取的User Prompt"""
    return f"请根据以上规则，从以下文本中提取SPO三元组:\n\n---\n{text_chunk}\n---"


def robust_json_parse(content_str):
    """增强型JSON解析，自动修正和提取LLM返回内容中的JSON片段。"""
    import json, re
    content = content_str.strip()
    # 去除markdown代码块
    if content.startswith('```json'):
        content = content[7:]
    if content.endswith('```'):
        content = content[:-3]
    content = content.strip()
    # 直接解析
    try:
        return json.loads(content)
    except Exception:
        pass
    # 替换单引号为双引号
    try:
        fixed = content.replace("'", '"')
        return json.loads(fixed)
    except Exception:
        pass
    # 正则提取第一个JSON对象或数组
    match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', content)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    # 兜底
    return None

def _call_llm_api(system_prompt: str, user_prompt: str, config: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """内部辅助函数，用于调用LLM API并获取SPO三元组。"""
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
    
    # 尝试两种API格式：带response_format和不带response_format
    data_with_format = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    
    data_without_format = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        # 首先尝试带response_format的请求
        try:
            response = requests.post(f"{api_base}/chat/completions", headers=headers, json=data_with_format, timeout=timeout)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 400:
                logger.warning("API不支持response_format，尝试不带格式的请求...")
                response = requests.post(f"{api_base}/chat/completions", headers=headers, json=data_without_format, timeout=timeout)
                response.raise_for_status()
            else:
                raise
        
        response_json = response.json()
        content_str = response_json['choices'][0]['message']['content']
        
        # 先判断是否为error响应
        if '"error"' in content_str.lower() or '\"error\"' in content_str.lower():
            logger.warning(f"LLM返回错误信息: {content_str[:200]}...")
            return None
        
        # 尝试多轮健壮解析
        result = robust_json_parse(content_str)
        if result is not None:
            return result
        
        logger.error(f"LLM返回内容无法解析: {content_str[:500]}...")
        return None
                
    except requests.exceptions.RequestException as e:
        logger.error(f"调用LLM API时发生网络错误: {e}")
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"解析LLM响应时出错: {e}")
        return None
    except Exception as e:
        logger.error(f"调用LLM API时出现未知错误: {e}")
        return None


def split_to_sentences(text: str) -> list:
    # 简单分句，支持中英文句号、问号、感叹号和换行
    sents = re.split(r'[。！？.!?\n]', text)
    return [s.strip() for s in sents if s.strip()]


def split_to_paragraphs(text: str) -> list:
    # 按空行或缩进分段
    import re
    paras = re.split(r'\n\s*\n|(?<=\n)\s{2,}', text)
    return [p.strip() for p in paras if p.strip()]


def extract_triplets_by_sentence(text: str, config: dict, max_workers: int = 4, retry_depth: int = 2) -> list:
    """
    先分句抽取，再全局合并去重。
    """
    sentences = split_to_sentences(text)
    from core.chunking import is_informative_chunk
    valid_sents = [s for s in sentences if is_informative_chunk(s)]
    logger = logging.getLogger(__name__)
    logger.info(f"分句后共 {len(valid_sents)} 个有效句子")
    # 句子级并发抽取
    all_triplets = []
    def process_sent(i, sent):
        user_prompt = get_extraction_user_prompt(sent)
        triplets = _call_llm_api(EXTRACTION_SYSTEM_PROMPT, user_prompt, config)
        if triplets:
            valid_triplets = [t for t in triplets if isinstance(t, dict) and all(k in t for k in ["subject", "predicate", "object"])]
            return valid_triplets
        return []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_sent, i, sent): i for i, sent in enumerate(valid_sents)}
        for future in as_completed(futures):
            result = future.result()
            all_triplets.extend(result)
    # 全局去重
    unique = {(t['subject'], t['predicate'], t['object']): t for t in all_triplets}
    logger.info(f"分句抽取后全局去重，最终三元组数: {len(unique)}")
    return list(unique.values())


def extract_triplets(text_chunks: List[str], config: Dict[str, Any], max_workers: int = 4, retry_depth: int = 2) -> List[Dict[str, str]]:
    """
    并发从文本块列表中抽取SPO三元组，失败块自动细分重试。
    """
    all_triplets = []
    logger = logging.getLogger(__name__)
    llm_conf = config.get('llm', {})
    min_chunk_tokens = llm_conf.get('min_chunk_tokens', 512)
    def process_chunk(i, chunk, depth=0):
        user_prompt = get_extraction_user_prompt(chunk)
        triplets = _call_llm_api(EXTRACTION_SYSTEM_PROMPT, user_prompt, config)
        if triplets:
            valid_triplets = [t for t in triplets if isinstance(t, dict) and all(k in t for k in ["subject", "predicate", "object"])]
            logger.info(f"块 {i+1} 成功抽取 {len(valid_triplets)} 个三元组。")
            return valid_triplets
        else:
            logger.warning(f"块 {i+1} 未能抽取到三元组。")
            # 失败重试：自动细分为更小块
            if depth < retry_depth and len(chunk) > 200:
                # 细分为更小块
                sub_config = dict(config)
                sub_config['llm'] = dict(llm_conf)
                sub_config['llm']['chunk_tokens'] = max(min_chunk_tokens, llm_conf.get('chunk_tokens', 2048)//2)
                from core.chunking import smart_chunk_text
                sub_chunks = smart_chunk_text(chunk, sub_config)
                results = []
                for sub in sub_chunks:
                    results.extend(process_chunk(i, sub, depth+1))
                return results
            return []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk, i, chunk): i for i, chunk in enumerate(text_chunks)}
        for future in as_completed(futures):
            result = future.result()
            all_triplets.extend(result)
    logger.info(f"SPO抽取完成，共获得 {len(all_triplets)} 个三元组。")
    return all_triplets


def extract_triplets_adaptive(text: str, config: dict, max_workers: int = 4, retry_depth: int = 2) -> list:
    from core.chunking import smart_chunk_text, is_informative_chunk
    # 先智能分块
    chunks = smart_chunk_text(text, config)
    all_chunks = []
    for chunk in chunks:
        # 块过大且包含多个句号时再分句
        if len(chunk) > 500 and (chunk.count('。') + chunk.count('.') > 2):
            sents = [s for s in split_to_sentences(chunk) if is_informative_chunk(s)]
            all_chunks.extend(sents)
        else:
            all_chunks.append(chunk)
    # 并发抽取+失败重试
    return extract_triplets(all_chunks, config, max_workers=max_workers, retry_depth=retry_depth)

# 分步抽取函数

def extract_stepwise(text: str, config: dict) -> dict:
    """
    先抽取成分、工艺、性能、组织结构，最后全局合并。
    增加了异常处理，避免单个抽取失败影响整个流程。
    """
    def call_simple_llm(prompt, text):
        user_prompt = text
        try:
            resp = _call_llm_api(prompt, user_prompt, config)
            if isinstance(resp, list):
                return [str(x).strip() for x in resp if str(x).strip()]
            if isinstance(resp, dict):
                # 兼容返回dict的情况
                return [str(x).strip() for v in resp.values() for x in (v if isinstance(v, list) else [v]) if str(x).strip()]
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"分步抽取时调用LLM失败: {e}")
        return []
    
    # 安全调用每个抽取步骤，失败时返回空列表
    try:
        components = call_simple_llm(COMPONENT_PROMPT, text)
    except Exception as e:
        logger.warning(f"成分抽取失败: {e}")
        components = []
    
    try:
        processes = call_simple_llm(PROCESS_PROMPT, text)
    except Exception as e:
        logger.warning(f"工艺抽取失败: {e}")
        processes = []
    
    try:
        performances = call_simple_llm(PERFORMANCE_PROMPT, text)
    except Exception as e:
        logger.warning(f"性能抽取失败: {e}")
        performances = []
    
    try:
        structures = call_simple_llm(STRUCTURE_PROMPT, text)
    except Exception as e:
        logger.warning(f"组织结构抽取失败: {e}")
        structures = []
    
    return {
        "components": components,
        "processes": processes,
        "performances": performances,
        "structures": structures
    }

def generate_candidate_spo(stepwise: dict) -> list:
    """
    基于分步抽取结果，规则法自动生成候选SPO三元组。
    只生成常见链式关系：成分-工艺-性能-组织结构。
    """
    candidates = []
    comps = stepwise.get('components', [])
    procs = stepwise.get('processes', [])
    perfs = stepwise.get('performances', [])
    structs = stepwise.get('structures', [])
    # 成分-影响-性能
    for c in comps:
        for p in perfs:
            candidates.append({'subject': c, 'predicate': 'influences', 'object': p})
    # 成分-影响-组织结构
    for c in comps:
        for s in structs:
            candidates.append({'subject': c, 'predicate': 'influences', 'object': s})
    # 工艺-影响-性能
    for proc in procs:
        for p in perfs:
            candidates.append({'subject': proc, 'predicate': 'influences', 'object': p})
    # 工艺-影响-组织结构
    for proc in procs:
        for s in structs:
            candidates.append({'subject': proc, 'predicate': 'influences', 'object': s})
    # 组织结构-影响-性能
    for s in structs:
        for p in perfs:
            candidates.append({'subject': s, 'predicate': 'influences', 'object': p})
    # 成分-工艺-性能-组织结构链式关系（可拆分为多条SPO）
    for c in comps:
        for proc in procs:
            for s in structs:
                for p in perfs:
                    candidates.append({'subject': c, 'predicate': 'influences', 'object': proc})
                    candidates.append({'subject': proc, 'predicate': 'influences', 'object': s})
                    candidates.append({'subject': s, 'predicate': 'influences', 'object': p})
    # 去重
    unique = {(t['subject'], t['predicate'], t['object']): t for t in candidates}
    return list(unique.values())

def generate_candidate_spo_local(text: str, config: dict) -> tuple:
    """
    只组合同一段落/句子内的实体，统计频次，返回候选SPO及频次dict。
    支持多线程并发处理段落，提升速度。
    """
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor, as_completed
    paragraphs = split_to_paragraphs(text)
    logger = logging.getLogger(__name__)
    logger.info(f"开始分段落抽取，共 {len(paragraphs)} 个段落")
    all_spo = []
    # 并发度可通过config设置，deepseek官方推荐最大并发为8
    max_workers = config.get('llm', {}).get('max_concurrency', 8)
    def process_para(para):
        try:
            if len(para.strip()) < 20:
                return []
            stepwise = extract_stepwise(para, config)
            return generate_candidate_spo(stepwise)
        except Exception as e:
            logger.warning(f"处理段落时出错: {e}")
            return []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_para, para): i for i, para in enumerate(paragraphs)}
        for idx, future in enumerate(as_completed(futures)):
            result = future.result()
            all_spo.extend(result)
            if (idx+1) % 5 == 0 or (idx+1) == len(paragraphs):
                logger.info(f"已处理 {idx+1}/{len(paragraphs)} 个段落")
    if not all_spo:
        logger.warning("未能抽取到任何候选SPO，返回空结果")
        return [], Counter()
    spo_counter = Counter((t['subject'], t['predicate'], t['object']) for t in all_spo)
    spo_list = [dict(subject=s, predicate=p, object=o, freq=spo_counter[(s,p,o)]) for (s,p,o) in spo_counter]
    logger.info(f"段落级抽取完成，共获得 {len(spo_list)} 个不重复的候选SPO")
    return spo_list, spo_counter

def confidence_partition(main_spo: list, candidate_spo: list) -> dict:
    """
    置信度分级：高=LLM主干SPO，中=规则法与主干重叠，低=规则法高频，极低=其他。
    """
    main_set = set((t['subject'], t['predicate'], t['object']) for t in main_spo)
    cand_set = set((t['subject'], t['predicate'], t['object']) for t in candidate_spo)
    freq_map = { (t['subject'], t['predicate'], t['object']): t.get('freq', 1) for t in candidate_spo }
    overlap = main_set & cand_set
    high = [t for t in main_spo]
    mid = [t for t in candidate_spo if (t['subject'], t['predicate'], t['object']) in overlap]
    midlow = [t for t in candidate_spo if freq_map.get((t['subject'], t['predicate'], t['object']), 1) > 1 and (t['subject'], t['predicate'], t['object']) not in overlap]
    low = [t for t in candidate_spo if freq_map.get((t['subject'], t['predicate'], t['object']), 1) == 1 and (t['subject'], t['predicate'], t['object']) not in overlap]
    return {'high': high, 'mid': mid, 'midlow': midlow, 'low': low}

# ====== 补充优化后的铝合金领域实体与关系词表 ======
ALUMINUM_ENTITY_TYPES = {
    "成分": [
        "li", "mg", "si", "al", "cu", "ni", "aln", "alnp", "al2culi", "al3li", "mg2si", "al2limg", "allisi",
        "al-12si", "al-4.5cu", "al-8alnp", "zl109", "al-12si-4cu-2ni-1mg"
    ],
    "工艺": [
        "铸态", "均热", "固溶", "时效", "预拉伸", "挤压", "热处理", "人工时效", "t1", "t4", "t6", "液-固原位反应", "热挤压", "熔铸", "扩散", "热暴露"
    ],
    "性能": [
        "密度", "强度", "弹性模量", "延伸率", "屈服强度", "抗拉强度", "硬度", "布氏硬度", "显微硬度", "压缩强度", "热膨胀性能", "失效率", "晶间腐蚀敏感性", "点蚀", "orowan强化", "热错配位错强化", "载荷传递", "网状强化"
    ],
    "组织结构": [
        "δ'-al3li", "β\"-mg2si", "pfzs", "晶粒", "析出相", "al2limg", "allisi", "t1相", "第二相", "纳米析出相", "微骨架", "半连续网状结构", "条带状结构", "颗粒状", "枝状", "孪晶", "晶界富cu层", "无析出区", "共晶区", "共晶si相"
    ],
    "腐蚀类型": ["晶间腐蚀", "点蚀"],
    "强化机制": ["orowan强化", "热错配位错强化", "载荷传递", "网状强化"]
}
ALUMINUM_RELATION_TYPES = [
    "包含", "析出", "强化", "影响", "提高", "降低", "促进", "抑制", "形成", "结合", "替换", "损失", "演变", "测试", "提升", "缩短", "细化", "分布", "增长", "达到", "诱导", "阻碍", "调控", "变质", "改善", "转变", "分布于", "依附", "协同强化", "承担载荷"
]

# ====== 伪关系检测增强（主宾类型细分、谓词过滤） ======
def is_pseudo_relation(triplet: dict, entity_types: dict = ALUMINUM_ENTITY_TYPES, rel_types: list = ALUMINUM_RELATION_TYPES) -> bool:
    subj, pred, obj = triplet.get("subject", "").lower(), triplet.get("predicate", "").lower(), triplet.get("object", "").lower()
    if not subj or not obj or not pred:
        return True
    if subj == obj:
        return True
    # 主宾类型相同或均为细分类（如腐蚀类型、强化机制、组织结构形貌等）
    def get_type(e):
        for t, lst in entity_types.items():
            if e in lst:
                return t
        return None
    subj_type, obj_type = get_type(subj), get_type(obj)
    if subj_type and obj_type and subj_type == obj_type:
        return True
    # 无谓词或谓词不在领域关系词表，或谓词为描述性非动词
    if pred not in rel_types or len(pred) < 2:
        return True
    # 其他可扩展伪关系规则
    return False

def detect_pseudo_relations(triplets: list, entity_types: dict = ALUMINUM_ENTITY_TYPES, rel_types: list = ALUMINUM_RELATION_TYPES) -> list:
    """
    批量检测伪三元组，返回伪三元组列表。
    """
    return [t for t in triplets if is_pseudo_relation(t, entity_types, rel_types)]

# ====== LLM主干抽取Prompt模板优化（细分类型、领域词表丰富） ======
def build_llm_prompt(text: str, stepwise_entities: dict) -> str:
    entity_types_str = "\n".join([
        f"- {k}：{', '.join(v)}" for k, v in ALUMINUM_ENTITY_TYPES.items()
    ])
    rel_types_str = ", ".join(ALUMINUM_RELATION_TYPES)
    stepwise_str = "\n".join([
        f"{k}：{', '.join(v)}" for k, v in stepwise_entities.items() if v
    ])
    prompt = f"""
请根据以下铝合金领域科研文献摘要，抽取所有高质量的三元组（SPO），每个三元组应包含：主语（实体）、谓语（关系）、宾语（实体），并标注主语和宾语的类型（如成分、工艺、性能、组织结构、腐蚀类型、强化机制等）。请结合下方领域词表和分步抽取结果，优先抽取与这些实体相关的主干SPO，避免无关、重复、主宾类型相同、无实际语义关系的三元组。

【实体类型示例】\n{entity_types_str}
【关系类型示例】\n{rel_types_str}
【分步抽取实体列表】\n{stepwise_str}
【摘要内容】\n{text}
【输出格式】\n[
  {{"subject": "...", "subject_type": "...", "predicate": "...", "object": "...", "object_type": "..."}},
  ...
]
"""
    return prompt

def extract_main_spo_with_context(text: str, stepwise_result: dict, config: dict) -> list:
    """
    基于分步抽取结果，构建上下文增强Prompt，调用LLM抽取主干SPO。
    """
    prompt = build_llm_prompt(text, stepwise_result)
    return _call_llm_api("", prompt, config) or []

# ====== 日志与报告输出增强（示例） ======
def log_extraction_report(triplets: list, pseudo_count: int, total: int, logger=logger):
    logger.info(f"抽取三元组总数: {total}, 伪关系数: {pseudo_count}, 伪关系率: {pseudo_count/total if total else 0:.2%}")
    # 可扩展输出到csv、json等

def export_for_annotation(triplets: list, file_path: str):
    """
    导出三元组供人工标注，格式为csv。
    """
    import csv
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['subject', 'predicate', 'object'])
        writer.writeheader()
        for t in triplets:
            writer.writerow({k: t.get(k, '') for k in ['subject', 'predicate', 'object']})

def log_extraction_stats(triplets: list, pseudo: list, logger=None):
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    logger.info(f"三元组总数: {len(triplets)}，伪关系数: {len(pseudo)}，伪关系率: {len(pseudo)/max(len(triplets),1):.2%}")

def evaluate_extraction(triplets: list, gold_standard: list, pseudo: list, logger=None, csv_path: str = None):
    """
    评估抽取质量，输出准确率、召回率、F1、伪关系率到日志和可选csv。
    gold_standard: 人工标注的三元组列表（dict）。
    pseudo: 伪三元组列表。
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    triplet_set = set((t['subject'], t['predicate'], t['object']) for t in triplets)
    gold_set = set((t['subject'], t['predicate'], t['object']) for t in gold_standard)
    correct = triplet_set & gold_set
    precision = len(correct) / len(triplet_set) if triplet_set else 0
    recall = len(correct) / len(gold_set) if gold_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    pseudo_rate = len(pseudo) / max(len(triplets), 1)
    logger.info(f"抽取评估：准确率={precision:.2%} 召回率={recall:.2%} F1={f1:.2%} 伪关系率={pseudo_rate:.2%} 正确三元组数={len(correct)} 标注三元组数={len(gold_set)}")
    if csv_path:
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['precision', 'recall', 'f1', 'pseudo_rate', 'correct', 'total_extracted', 'total_gold'])
            writer.writeheader()
            writer.writerow({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'pseudo_rate': pseudo_rate,
                'correct': len(correct),
                'total_extracted': len(triplet_set),
                'total_gold': len(gold_set)
            })
    return {'precision': precision, 'recall': recall, 'f1': f1, 'pseudo_rate': pseudo_rate, 'correct': len(correct)}