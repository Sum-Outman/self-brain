# Copyright 2025 AGI System Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 知识库初始化脚本 - 预加载所有领域的基礎知识
# Knowledge Base Initialization Script - Preload basic knowledge for all domains

import json
import logging
from pathlib import Path
from py2neo import Graph, Node, Relationship

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeInitializer:
    def __init__(self, neo4j_uri='bolt://localhost:7687', neo4j_user='neo4j', neo4j_password='password'):
        self.driver = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # 15个知识领域的映射
        self.domain_mappings = {
            'physics': {'en': 'Physics', 'zh': '物理'},
            'mathematics': {'en': 'Mathematics', 'zh': '数学'},
            'chemistry': {'en': 'Chemistry', 'zh': '化学'},
            'medicine': {'en': 'Medicine', 'zh': '医学'},
            'law': {'en': 'Law', 'zh': '法学'},
            'history': {'en': 'History', 'zh': '历史'},
            'sociology': {'en': 'Sociology', 'zh': '社会学'},
            'humanities': {'en': 'Humanities', 'zh': '人文学'},
            'psychology': {'en': 'Psychology', 'zh': '心理学'},
            'economics': {'en': 'Economics', 'zh': '经济学'},
            'management': {'en': 'Management', 'zh': '管理学'},
            'mechanical_engineering': {'en': 'Mechanical Engineering', 'zh': '机械工程'},
            'electronic_engineering': {'en': 'Electronic Engineering', 'zh': '电子工程'},
            'food_engineering': {'en': 'Food Engineering', 'zh': '食品工程'},
            'chemical_engineering': {'en': 'Chemical Engineering', 'zh': '化学工程'}
        }
    
    def clear_existing_knowledge(self):
        """清空现有知识图谱"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("成功清空现有知识图谱 | Successfully cleared existing knowledge graph")
            return True
        except Exception as e:
            logger.error(f"清空知识图谱错误: {str(e)} | Clear knowledge graph error: {str(e)}")
            return False
    
    def build_domain_structure(self):
        """构建领域结构"""
        try:
            with self.driver.session() as session:
                # 创建所有领域节点
                for domain_key, translations in self.domain_mappings.items():
                    session.run(
                        "MERGE (d:Domain {key: $key, name: $name, name_zh: $name_zh})",
                        key=domain_key,
                        name=translations['en'],
                        name_zh=translations['zh']
                    )
                
                # 创建关系类型
                session.run("""
                    MERGE (r:RelationType {type: 'HAS_PROPERTY'})
                    MERGE (r2:RelationType {type: 'RELATED_TO'})
                    MERGE (r3:RelationType {type: 'SUBCLASS_OF'})
                    MERGE (r4:RelationType {type: 'PART_OF'})
                    MERGE (r5:RelationType {type: 'USES'})
                    MERGE (r6:RelationType {type: 'DEPENDS_ON'})
                """)
            
            logger.info("成功构建领域结构 | Successfully built domain structure")
            return True
        except Exception as e:
            logger.error(f"构建领域结构错误: {str(e)} | Build domain structure error: {str(e)}")
            return False
    
    def load_physics_knowledge(self):
        """加载物理知识"""
        physics_concepts = [
            {
                'concept': 'Newton\'s Laws of Motion',
                'concept_zh': '牛顿运动定律',
                'description': 'Three physical laws that form the foundation for classical mechanics',
                'description_zh': '形成经典力学基础的三个物理定律',
                'properties': {
                    'first_law': 'An object at rest remains at rest, and an object in motion remains in motion at constant speed and in a straight line unless acted on by an unbalanced force.',
                    'second_law': 'The acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass.',
                    'third_law': 'For every action, there is an equal and opposite reaction.'
                }
            },
            {
                'concept': 'Quantum Mechanics',
                'concept_zh': '量子力学',
                'description': 'Fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles',
                'description_zh': '物理学中的基础理论，描述了原子和亚原子粒子尺度上自然的物理性质',
                'properties': {
                    'wave-particle_duality': 'Quantum objects exhibit both wave-like and particle-like properties',
                    'uncertainty_principle': 'It is impossible to simultaneously know both the position and momentum of a particle with perfect accuracy'
                }
            }
        ]
        
        return self._load_domain_knowledge('physics', physics_concepts)
    
    def load_mathematics_knowledge(self):
        """加载数学知识"""
        math_concepts = [
            {
                'concept': 'Calculus',
                'concept_zh': '微积分',
                'description': 'Branch of mathematics focused on limits, functions, derivatives, integrals, and infinite series',
                'description_zh': '数学分支，专注于极限、函数、导数、积分和无穷级数',
                'properties': {
                    'differential_calculus': 'Studies rates of change and slopes of curves',
                    'integral_calculus': 'Studies accumulation of quantities and areas under curves'
                }
            },
            {
                'concept': 'Linear Algebra',
                'concept_zh': '线性代数',
                'description': 'Branch of mathematics concerning linear equations, linear functions, and their representations through matrices and vector spaces',
                'description_zh': '数学分支，涉及线性方程、线性函数及其通过矩阵和向量空间的表示',
                'properties': {
                    'vectors': 'Quantities having both magnitude and direction',
                    'matrices': 'Rectangular arrays of numbers arranged in rows and columns'
                }
            }
        ]
        
        return self._load_domain_knowledge('mathematics', math_concepts)
    
    def load_chemistry_knowledge(self):
        """加载化学知识"""
        chemistry_concepts = [
            {
                'concept': 'Periodic Table',
                'concept_zh': '元素周期表',
                'description': 'Tabular arrangement of the chemical elements, organized by atomic number, electron configuration, and recurring chemical properties',
                'description_zh': '化学元素的表格排列，按原子序数、电子配置和重复化学性质组织',
                'properties': {
                    'groups': 'Vertical columns in the periodic table',
                    'periods': 'Horizontal rows in the periodic table'
                }
            }
        ]
        
        return self._load_domain_knowledge('chemistry', chemistry_concepts)
    
    def _load_domain_knowledge(self, domain_key, concepts):
        """加载特定领域的知识"""
        try:
            with self.driver.session() as session:
                for concept_data in concepts:
                    # 创建概念节点
                    session.run("""
                        MERGE (c:Concept {key: $key, name: $name, name_zh: $name_zh, description: $description, description_zh: $description_zh})
                    """, key=concept_data['concept'].lower().replace(' ', '_'),
                               name=concept_data['concept'],
                               name_zh=concept_data['concept_zh'],
                               description=concept_data['description'],
                               description_zh=concept_data['description_zh'])
                    
                    # 建立领域与概念的关系
                    session.run("""
                        MATCH (d:Domain {key: $domain}), (c:Concept {key: $concept})
                        MERGE (d)-[:CONTAINS]->(c)
                    """, domain=domain_key, concept=concept_data['concept'].lower().replace(' ', '_'))
                    
                    # 添加属性
                    if 'properties' in concept_data:
                        for prop_name, prop_value in concept_data['properties'].items():
                            session.run("""
                                MATCH (c:Concept {key: $concept})
                                MERGE (p:Property {name: $prop_name, value: $prop_value})
                                MERGE (c)-[:HAS_PROPERTY]->(p)
                            """, concept=concept_data['concept'].lower().replace(' ', '_'),
                                      prop_name=prop_name, prop_value=prop_value)
            
            logger.info(f"成功加载{domain_key}领域知识 | Successfully loaded {domain_key} domain knowledge")
            return True
        except Exception as e:
            logger.error(f"加载{domain_key}知识错误: {str(e)} | Load {domain_key} knowledge error: {str(e)}")
            return False
    
    def initialize_all_domains(self):
        """初始化所有领域的知识"""
        logger.info("开始初始化知识库 | Starting knowledge base initialization")
        
        # 清空现有知识
        if not self.clear_existing_knowledge():
            return False
        
        # 构建领域结构
        if not self.build_domain_structure():
            return False
        
        # 加载各领域知识
        domains_to_load = [
            self.load_physics_knowledge,
            self.load_mathematics_knowledge,
            self.load_chemistry_knowledge,
            # 可以继续添加其他领域
        ]
        
        success_count = 0
        for load_func in domains_to_load:
            if load_func():
                success_count += 1
        
        logger.info(f"知识库初始化完成 | Knowledge base initialization completed")
        logger.info(f"成功加载 {success_count}/{len(domains_to_load)} 个领域 | Successfully loaded {success_count}/{len(domains_to_load)} domains")
        
        return success_count == len(domains_to_load)

def main():
    """主函数"""
    initializer = KnowledgeInitializer()
    
    if initializer.initialize_all_domains():
        print("知识库初始化成功 | Knowledge base initialized successfully")
        return 0
    else:
        print("知识库初始化失败 | Knowledge base initialization failed")
        return 1

if __name__ == "__main__":
    exit(main())
