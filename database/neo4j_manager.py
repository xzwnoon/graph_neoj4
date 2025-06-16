"""
模块: database.neo4j_manager
描述: 管理与Neo4j数据库的连接和操作。
"""
import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError, ServiceUnavailable

logger = logging.getLogger(__name__)

class Neo4jManager:
    """
    一个用于管理Neo4j数据库交互的类，包括连接、关闭、创建约束和批量数据导入。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Neo4jManager。

        Args:
            config (Dict[str, Any]): 包含 'uri', 'user', 'password' 的Neo4j配置字典。
        """
        db_config = config.get("neo4j", {})
        self._uri = db_config.get("uri")
        self._user = db_config.get("user")
        self._password = db_config.get("password")
        self._driver: Optional[Driver] = None

        if not all([self._uri, self._user, self._password]):
            raise ValueError("Neo4j配置不完整，必须提供uri, user, 和 password。")
        
        self.connect()

    def connect(self) -> None:
        """建立与数据库的连接驱动。"""
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            self._driver.verify_connectivity()
            logger.info(f"成功连接到 Neo4j 数据库 at {self._uri}")
        except (ServiceUnavailable, Neo4jError) as e:
            logger.error(f"无法连接到 Neo4j 数据库: {e}", exc_info=True)
            self._driver = None
            raise  # 重新抛出异常，让调用者知道连接失败

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._driver is not None:
            self._driver.close()
            logger.info("Neo4j 数据库连接已关闭。")

    def create_constraints(self) -> None:
        """在数据库中创建唯一性约束，确保数据模型的完整性。"""
        if not self._driver:
            logger.error("数据库未连接，无法创建约束。")
            return
            
        # 为实体节点创建一个唯一性约束，确保每个标准化的实体只存在一次
        constraint_query = "CREATE CONSTRAINT entity_unique_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        try:
            with self._driver.session() as session:
                session.run(constraint_query)
            logger.info("成功创建或确认 'Entity' 节点的唯一性约束。")
        except Neo4jError as e:
            logger.error(f"创建约束时出错: {e}", exc_info=True)

    def batch_import_data(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> None:
        """
        使用高效的批量操作将实体和关系导入Neo4j。

        Args:
            entities (List[Dict[str, Any]]): 实体字典列表，每个字典包含 'name' 和 'type'。
            relationships (List[Dict[str, Any]]): 关系字典列表，包含 'source', 'target', 'type'。
        """
        if not self._driver:
            logger.error("数据库未连接，无法导入数据。")
            return

        # 使用 UNWIND 和 MERGE 进行高效的批量导入
        # MERGE 会查找匹配的节点/关系，如果不存在则创建，避免重复
        import_entities_query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {name: entity.name})
        ON CREATE SET e.type = entity.type, e.created = timestamp()
        """
        
        import_relationships_query = """
        UNWIND $relationships AS rel
        MATCH (source:Entity {name: rel.source})
        MATCH (target:Entity {name: rel.target})
        MERGE (source)-[r:RELATIONSHIP {type: rel.type}]->(target)
        ON CREATE SET r.created = timestamp()
        """

        try:
            with self._driver.session() as session:
                logger.info(f"开始导入 {len(entities)} 个实体...")
                session.run(import_entities_query, entities=entities)
                logger.info("实体导入完成。")

                logger.info(f"开始导入 {len(relationships)} 个关系...")
                session.run(import_relationships_query, relationships=relationships)
                logger.info("关系导入完成。")
        except Neo4jError as e:
            logger.error(f"批量导入数据时出错: {e}", exc_info=True)
            raise