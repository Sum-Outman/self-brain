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
# 简化版知识库初始化脚本 - 不使用Neo4j，使用JSON文件存储
# Simplified Knowledge Base Initialization Script - Uses JSON file instead of Neo4j

import json
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleKnowledgeInitializer:
    def __init__(self, data_file='knowledge_base.json'):
        self.data_file = Path(data_file)
        self.knowledge_base = {
            'domains': {},
            'concepts': {},
            'relations': []
        }
        
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
    
    def load_existing_knowledge(self):
        """加载现有的知识库数据"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                logger.info("成功加载现有知识库 | Successfully loaded existing knowledge base")
                return True
            except Exception as e:
                logger.error(f"加载知识库错误: {str(e)} | Load knowledge base error: {str(e)}")
        return False
    
    def save_knowledge_base(self):
        """保存知识库到文件"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
            logger.info("成功保存知识库 | Successfully saved knowledge base")
            return True
        except Exception as e:
            logger.error(f"保存知识库错误: {str(e)} | Save knowledge base error: {str(e)}")
            return False
    
    def build_domain_structure(self):
        """构建领域结构"""
        try:
            for domain_key, translations in self.domain_mappings.items():
                self.knowledge_base['domains'][domain_key] = {
                    'name_en': translations['en'],
                    'name_zh': translations['zh'],
                    'concepts': []
                }
            
            logger.info("成功构建领域结构 | Successfully built domain structure")
            return True
        except Exception as e:
            logger.error(f"构建领域结构错误: {str(e)} | Build domain structure error: {str(e)}")
            return False
    
    def load_physics_knowledge(self):
        """加载物理知识"""
        physics_concepts = [
            {
                'id': 'newtons_laws',
                'name_en': "Newton's Laws of Motion",
                'name_zh': '牛顿运动定律',
                'description_en': 'Three physical laws that form the foundation for classical mechanics',
                'description_zh': '形成经典力学基础的三个物理定律',
                'properties': {
                    'first_law': 'An object at rest remains at rest, and an object in motion remains in motion at constant speed and in a straight line unless acted on by an unbalanced force.',
                    'second_law': 'The acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass.',
                    'third_law': 'For every action, there is an equal and opposite reaction.'
                }
            },
            {
                'id': 'quantum_mechanics',
                'name_en': 'Quantum Mechanics',
                'name_zh': '量子力学',
                'description_en': 'Fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles',
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
                'id': 'calculus',
                'name_en': 'Calculus',
                'name_zh': '微积分',
                'description_en': 'Branch of mathematics focused on limits, functions, derivatives, integrals, and infinite series',
                'description_zh': '数学分支，专注于极限、函数、导数、积分和无穷级数',
                'properties': {
                    'differential_calculus': 'Studies rates of change and slopes of curves',
                    'integral_calculus': 'Studies accumulation of quantities and areas under curves'
                }
            },
            {
                'id': 'linear_algebra',
                'name_en': 'Linear Algebra',
                'name_zh': '线性代数',
                'description_en': 'Branch of mathematics concerning linear equations, linear functions, and their representations through matrices and vector spaces',
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
                'id': 'periodic_table',
                'name_en': 'Periodic Table',
                'name_zh': '元素周期表',
                'description_en': 'Tabular arrangement of the chemical elements, organized by atomic number, electron configuration, and recurring chemical properties',
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
            for concept_data in concepts:
                # 添加到概念库
                self.knowledge_base['concepts'][concept_data['id']] = concept_data
                
                # 添加到领域的概念列表
                if domain_key not in self.knowledge_base['domains']:
                    self.knowledge_base['domains'][domain_key] = {'concepts': []}
                
                self.knowledge_base['domains'][domain_key]['concepts'].append(concept_data['id'])
            
            logger.info(f"成功加载{domain_key}领域知识 | Successfully loaded {domain_key} domain knowledge")
            return True
        except Exception as e:
            logger.error(f"加载{domain_key}知识错误: {str(e)} | Load {domain_key} knowledge error: {str(e)}")
            return False
    
    def initialize_all_domains(self):
        """初始化所有领域的知识"""
        logger.info("开始初始化知识库 | Starting knowledge base initialization")
        
        # 加载现有知识（如果有）
        self.load_existing_knowledge()
        
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
        
        # 保存知识库
        if not self.save_knowledge_base():
            return False
        
        logger.info(f"知识库初始化完成 | Knowledge base initialization completed")
        logger.info(f"成功加载 {success_count}/{len(domains_to_load)} 个领域 | Successfully loaded {success_count}/{len(domains_to_load)} domains")
        
        return success_count > 0

def main():
    """主函数"""
    initializer = SimpleKnowledgeInitializer()
    
    if initializer.initialize_all_domains():
        print("知识库初始化成功 | Knowledge base initialized successfully")
        print("知识库已保存到: knowledge_base.json")
        return 0
    else:
        print("知识库初始化失败 | Knowledge base initialization failed")
        return 1

if __name__ == "__main__":
    exit(main())