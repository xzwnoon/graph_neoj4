"""
主模块 (main.py)
描述: 协调整个知识图谱构建流程的入口点。
"""
import logging
import os
import csv
import json
from typing import Dict, List, Any, Optional

# 导入项目模块
from config.settings import get_settings
from utils.logging_config import setup_logging
from utils.file_handler import read_text_file
from core.chunking import smart_chunk_text
from core.extraction import extract_triplets_adaptive, extract_stepwise, generate_candidate_spo, detect_pseudo_relations, log_extraction_stats, export_for_annotation, generate_candidate_spo_local, confidence_partition, extract_main_spo_with_context
from core.standardization import standardize_entities
from core.inference import infer_new_relations
from database.neo4j_manager import Neo4jManager


def transform_data_for_db(
    triplets: List[Dict[str, str]],
    standardization_map: Dict[str, str],
    inferred_relations: List[Dict[str, str]]
) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    """
    将处理后的数据转换为适合数据库导入的格式。

    Args:
        triplets: 原始SPO三元组。
        standardization_map: 实体标准化映射。
        inferred_relations: 推理出的新关系。

    Returns:
        A tuple containing (list of entity dicts, list of relationship dicts).
    """
    entities = {}
    relationships = []

    all_relations = triplets + inferred_relations

    for rel in all_relations:
        # 使用标准化后的实体名称
        source_name = standardization_map.get(rel['subject'], rel['subject'])
        target_name = standardization_map.get(rel['object'], rel['object'])
        
        # 添加实体到字典，使用字典去重
        if source_name not in entities:
            entities[source_name] = {"name": source_name, "type": "Entity"}
        if target_name not in entities:
            entities[target_name] = {"name": target_name, "type": "Entity"}
        
        # 添加关系
        relationships.append({
            "source": source_name,
            "target": target_name,
            "type": rel['predicate']
        })
    
    return list(entities.values()), relationships


def assign_confidence(main_spo, candidate_spo, config=None):
    """为每个SPO分配置信度分数，支持从config读取区间配置。"""
    if config is not None and 'confidence_rules' in config:
        rules = config['confidence_rules']
        main_val = rules.get('main_spo', 1.0)
        overlap_val = rules.get('overlap', 0.8)
        high_freq_base = rules.get('high_freq_base', 0.5)
        high_freq_step = rules.get('high_freq_step', 0.1)
        high_freq_max = rules.get('high_freq_max', 0.8)
        low_freq_val = rules.get('low_freq', 0.3)
    else:
        main_val = 1.0
        overlap_val = 0.8
        high_freq_base = 0.5
        high_freq_step = 0.1
        high_freq_max = 0.8
        low_freq_val = 0.3
    freq_map = { (t['subject'], t['predicate'], t['object']): t.get('freq', 1) for t in candidate_spo }
    main_set = set((t['subject'], t['predicate'], t['object']) for t in main_spo)
    cand_set = set((t['subject'], t['predicate'], t['object']) for t in candidate_spo)
    overlap = main_set & cand_set
    result = []
    for t in main_spo:
        t = t.copy()
        t['confidence'] = main_val
        result.append(t)
    for t in candidate_spo:
        key = (t['subject'], t['predicate'], t['object'])
        if key in main_set:
            continue  # 已在高置信度
        t = t.copy()
        freq = freq_map.get(key, 1)
        if key in overlap:
            t['confidence'] = overlap_val
        elif freq > 1:
            t['confidence'] = min(high_freq_base + high_freq_step * (freq - 2), high_freq_max)
        else:
            t['confidence'] = low_freq_val
        result.append(t)
    return result


def run_pipeline(input_file_path: str) -> Dict[str, Any]:
    """
    执行完整的知识图谱构建和存储流水线。

    Args:
        input_file_path (str): 输入的文本文件路径。

    Returns:
        Dict[str, Any]: 统计信息，包括三元组数、实体数、关系数。
    """
    logger = logging.getLogger(__name__)
    logger.info("--- 知识图谱构建流水线开始 ---")

    # 1. 加载配置
    config = get_settings()
    # 2. 读取输入文件
    text_content = read_text_file(input_file_path)
    if not text_content:
        logger.error("无法读取文件内容，流水线终止。")
        return {"triplets": 0, "entities": 0, "relationships": 0, "error": "读取失败"}
    # 分步抽取（成分、工艺、性能、组织结构）
    stepwise_result = extract_stepwise(text_content, config)
    logger.info(f"分步抽取结果: 成分: {stepwise_result['components']} 工艺: {stepwise_result['processes']} 性能: {stepwise_result['performances']} 组织结构: {stepwise_result['structures']}")
    # LLM增强Prompt抽取主干关系
    main_spo = extract_main_spo_with_context(text_content, stepwise_result, config)
    logger.info(f"LLM增强Prompt抽取主干SPO数: {len(main_spo)}")
    # 分段/分句内规则法生成候选SPO及频次
    candidate_spo, spo_counter = generate_candidate_spo_local(text_content, config)
    logger.info(f"分段/分句内规则法候选SPO数: {len(candidate_spo)}")
    # 置信度分级
    conf_parts = confidence_partition(main_spo, candidate_spo)
    logger.info(f"高置信度SPO: {len(conf_parts['high'])}，中置信度: {len(conf_parts['mid'])}，中低置信度: {len(conf_parts['midlow'])}，低置信度: {len(conf_parts['low'])}")
    # 只入库高/中置信度SPO
    all_spo = conf_parts['high'] + conf_parts['mid']
    unique_spo = {(t['subject'], t['predicate'], t['object']): t for t in all_spo}.values()
    logger.info(f"最终入库三元组数: {len(unique_spo)}")
    # 置信度分配与统计
    all_with_conf = assign_confidence(main_spo, candidate_spo, config)
    # 只选高+中置信度SPO
    high_mid_conf_spo = [t for t in all_with_conf if t['confidence'] >= 0.8]
    logger.info(f"高+中置信度SPO数: {len(high_mid_conf_spo)}")
    # 后续入库、伪关系检测、统计等均用high_mid_conf_spo
    pseudo = detect_pseudo_relations(high_mid_conf_spo)
    log_extraction_stats(high_mid_conf_spo, pseudo, logger)
    export_for_annotation([t for t in all_with_conf if t['confidence'] < 0.8], 'annotation_export.csv')
    # 5. 实体标准化
    standardization_map = standardize_entities(high_mid_conf_spo, config)
    # 6. 关系推理 (占位符)
    inferred_relations = infer_new_relations(high_mid_conf_spo, standardization_map, config)
    # 7. 数据转换
    entities_for_db, relationships_for_db = transform_data_for_db(high_mid_conf_spo, standardization_map, inferred_relations)
    logger.info(f"数据转换完成: {len(entities_for_db)} 个实体, {len(relationships_for_db)} 个关系。")

    # 8. 持久化到 Neo4j
    try:
        neo4j_manager = Neo4jManager(config)
        neo4j_manager.create_constraints()
        neo4j_manager.batch_import_data(entities_for_db, relationships_for_db)
        neo4j_manager.close()
        logger.info("数据成功持久化到 Neo4j 数据库。")
    except Exception as e:
        logger.critical(f"连接或写入 Neo4j 数据库时发生严重错误: {e}", exc_info=True)
        logger.critical("请检查 'config/settings.py' 中的数据库凭据和 Neo4j 服务状态。")
        return {"triplets": len(unique_spo), "entities": len(entities_for_db), "relationships": len(relationships_for_db), "error": str(e)}
    
    logger.info("--- 知识图谱构建流水线成功完成 ---")
    return {
        "triplets": len(unique_spo),
        "entities": len(entities_for_db),
        "relationships": len(relationships_for_db),
        "stepwise": stepwise_result,
        "pseudo_relations": len(pseudo),
        "error": None
    }


def is_valid_txt_file(file_path: str) -> bool:
    """判断是否为有效的txt文件（非空、非隐藏）"""
    return (
        file_path.lower().endswith('.txt')
        and os.path.isfile(file_path)
        and not os.path.basename(file_path).startswith('.')
        and os.path.getsize(file_path) > 0
    )


def find_txt_files(folder_path: str) -> List[str]:
    """递归查找所有有效txt文件"""
    txt_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            file_path = os.path.join(root, f)
            if is_valid_txt_file(file_path):
                txt_files.append(file_path)
    return txt_files


def batch_process_txt_files(folder_path: str, export_report: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    批量处理指定文件夹下所有txt文件，并输出处理日志和汇总报告。
    支持递归、进度提示、导出csv/json报告。
    """
    logger = logging.getLogger(__name__)
    txt_files = find_txt_files(folder_path)
    total = len(txt_files)
    report = []
    logger.info(f'共发现 {total} 个txt文件，开始批量处理...')
    for idx, file_path in enumerate(txt_files, 1):
        logger.info(f'[{idx}/{total}] 开始处理文件: {file_path}')
        file_report = {"file": file_path}
        try:
            stats = run_pipeline(file_path)
            file_report.update(stats or {})
            file_report["status"] = "成功" if not stats.get("error") else f"失败: {stats.get('error')}"
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}", exc_info=True)
            file_report["status"] = f"失败: {e}"
        report.append(file_report)
    logger.info("===== 批量处理汇总报告 =====")
    for item in report:
        logger.info(f"{item['file']} - 状态: {item['status']} - 实体: {item.get('entities', 0)} - 三元组: {item.get('triplets', 0)} - 关系: {item.get('relationships', 0)}")
    # 导出报告
    if export_report:
        if export_report.lower().endswith('.csv'):
            # 自动收集所有字段，保证所有结果都能导出
            all_fields = set()
            for item in report:
                all_fields.update(item.keys())
            fieldnames = list(all_fields)
            with open(export_report, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(report)
        elif export_report.lower().endswith('.json'):
            with open(export_report, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"处理报告已导出到: {export_report}")
    return report


if __name__ == "__main__":
    # 配置日志
    setup_logging(get_settings().get("logging", {}))

    # 批量处理指定文件夹下所有txt文件
    TXT_FOLDER = r"E:\github_git\pdf_graph\txt"
    REPORT_FILE = "batch_report.csv"  # 可选：导出为csv或json
    batch_process_txt_files(TXT_FOLDER, export_report=REPORT_FILE)
