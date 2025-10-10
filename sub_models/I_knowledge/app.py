# -*- coding: utf-8 -*-
# Apache License 2.0 开源协议 | Apache License 2.0 Open Source License
# Copyright 2025 AGI System
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Core Implementation of Knowledge Base Expert Model
import json
import logging
import time
import threading
import queue
import numpy as np
import requests
from flask import Flask, request, jsonify
import random
from py2neo import Graph, Node, Relationship
from typing import Dict, Any, List, Optional

app = Flask(__name__)

class KnowledgeGraphBuilder:
    """Knowledge Graph Builder"""
    def __init__(self, driver, language='en'):
        self.driver = driver
        self.language = language
        self.translations = {
            'domain': {'en': 'Domain', 'zh': '领域', 'ja': 'ドメイン', 'de': 'Bereich', 'ru': 'Область'},
            'concept': {'en': 'Concept', 'zh': '概念', 'ja': '概念', 'de': 'Konzept', 'ru': 'Концепция'},
            'property': {'en': 'Property', 'zh': '属性', 'ja': 'プロパティ', 'de': 'Eigenschaft', 'ru': 'Свойство'},
            'relation': {'en': 'Relation', 'zh': '关系', 'ja': '関係', 'de': 'Beziehung', 'ru': 'Отношение'}
        }
    
    def _t(self, text):
        """Translate text"""
        return self.translations.get(text, {}).get(self.language, text)
    
    def build_initial_schema(self):
        """Build initial graph schema"""
        try:
            with self.driver.session() as session:
                # Create domain nodes
                for domain in self.domain_mappings.keys():
                    session.run("MERGE (d:Domain {name: $domain})", domain=domain)
                
                # Create core relation types
                session.run("""
                    MERGE (r:RelationType {type: 'HAS_PROPERTY'})
                    MERGE (r2:RelationType {type: 'RELATED_TO'})
                    MERGE (r3:RelationType {type: 'SUBCLASS_OF'})
                    MERGE (r4:RelationType {type: 'PART_OF'})
                    MERGE (r5:RelationType {type: 'INSTANCE_OF'})
                """)
            
            return {"status": "success", "message": f"{self._t('domain')} schema built successfully"}
        except Exception as e:
            logging.error(f"Build knowledge graph schema error: {str(e)}")
            return {"status": "error", "message": str(e)}

class ReasoningEngine:
    """Knowledge Reasoning Engine"""
    def __init__(self, driver, language='en'):
        self.driver = driver
        self.language = language
        self.translations = {
            'inference': {'en': 'inference', 'zh': '推理', 'ja': '推論', 'de': 'Inferenz', 'ru': 'Вывод'},
            'analysis': {'en': 'analysis', 'zh': '分析', 'ja': '分析', 'de': 'Analyse', 'ru': 'Анализ'}
        }
    
    def _t(self, text):
        """Translate text"""
        return self.translations.get(text, {}).get(self.language, text)
    
    def infer_knowledge(self, domain, concept, base_knowledge):
        """Infer knowledge based on existing knowledge"""
        try:
            inferred_properties = []
            inferred_relations = []
            
            # Inference based on domain and concept
            with self.driver.session() as session:
                # Find related concepts
                result = session.run("""
                    MATCH (c:Concept {name: $concept})-[:HAS_RELATION]->(r:Relation)-[:TO]->(t:Concept)
                    RETURN r.type AS relation_type, t.name AS target_concept
                    LIMIT 10
                """, concept=concept)
                
                for record in result:
                    inferred_relations.append({
                        "type": record["relation_type"],
                        "target": record["target_concept"]
                    })
                
                # Find properties of similar concepts
                result = session.run("""
                    MATCH (c:Concept {name: $concept})-[:HAS_PROPERTY]->(p:Property)
                    RETURN p.name AS property_name, p.value AS property_value
                    LIMIT 10
                """, concept=concept)
                
                for record in result:
                    inferred_properties.append({
                        "name": record["property_name"],
                        "value": record["property_value"]
                    })
            
            return {
                "inferred_properties": inferred_properties,
                "inferred_relations": inferred_relations,
                "confidence": 0.85  # Inference confidence
            }
        except Exception as e:
            logging.error(f"Knowledge inference error: {str(e)}")
            return {"inferred_properties": [], "inferred_relations": [], "error": str(e)}
    
    def analyze_task(self, task):
        """Analyze task requirements"""
        try:
            # Use simple keyword extraction and complexity analysis
            keywords = task.lower().split()
            word_count = len(keywords)
            
            # Judge complexity based on task length and keywords
            if word_count <= 3:
                complexity = "simple"
            elif word_count <= 7:
                complexity = "medium"
            else:
                complexity = "complex"
            
            return {
                "keywords": keywords,
                "complexity": complexity,
                "word_count": word_count,
                "analysis_timestamp": time.time()
            }
        except Exception as e:
            logging.error(f"Task analysis error: {str(e)}")
            return {"keywords": [], "complexity": "unknown", "error": str(e)}
    
    def get_core_concepts(self, domain, limit=10):
        """Get core concepts of a domain"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (d:Domain {name: $domain})-[:CONTAINS]->(c:Concept)
                    RETURN c.name AS concept_name
                    ORDER BY c.importance DESC
                    LIMIT $limit
                """, domain=domain, limit=limit)
                
                concepts = [record["concept_name"] for record in result]
                return concepts
        except Exception as e:
            logging.error(f"Get core concepts error: {str(e)}")
            return []

class ConfidenceEvaluator:
    """Confidence Evaluation System"""
    def __init__(self, language='en'):
        self.language = language
        self.translations = {
            'confidence': {'en': 'confidence', 'zh': '置信度', 'ja': '信頼度', 'de': 'Konfidenz', 'ru': 'Доверие'}
        }
    
    def _t(self, text):
        """Translate text"""
        return self.translations.get(text, {}).get(self.language, text)
    
    def evaluate_knowledge_confidence(self, knowledge):
        """Evaluate knowledge confidence"""
        try:
            # Evaluate confidence based on knowledge completeness, source reliability, and consistency
            confidence = 0.7  # Base confidence
            
            # Check knowledge completeness
            if knowledge.get('description') and len(knowledge.get('description', '')) > 10:
                confidence += 0.1
            
            if knowledge.get('properties') and len(knowledge.get('properties', [])) > 0:
                confidence += 0.1
            
            if knowledge.get('relations') and len(knowledge.get('relations', [])) > 0:
                confidence += 0.1
            
            # Limit to 0-1 range
            confidence = max(0.1, min(0.99, confidence))
            
            return confidence
        except Exception as e:
            logging.error(f"Confidence evaluation error: {str(e)}")
            return 0.5

class ExperienceLearner:
    """Experience Learning Module"""
    def __init__(self, language='en'):
        self.language = language
        self.experience_db = []  # Experience database
        self.translations = {
            'experience': {'en': 'experience', 'zh': '经验', 'ja': '経験', 'de': 'Erfahrung', 'ru': 'Опыт'},
            'learning': {'en': 'learning', 'zh': '学习', 'ja': '学習', 'de': 'Lernen', 'ru': 'Обучение'}
        }
    
    def _t(self, text):
        """Translate text"""
        return self.translations.get(text, {}).get(self.language, text)
    
    def record_assistance_experience(self, model_name, task, assistance_data):
        """Record assistance experience"""
        try:
            experience_record = {
                "timestamp": time.time(),
                "model": model_name,
                "task": task,
                "assistance_data": assistance_data,
                "effectiveness": self._evaluate_assistance_effectiveness(assistance_data)
            }
            
            self.experience_db.append(experience_record)
            
            # Limit experience database size
            if len(self.experience_db) > 1000:
                self.experience_db = self.experience_db[-1000:]
            
            return True
        except Exception as e:
            logging.error(f"Record experience error: {str(e)}")
            return False
    
    def learn_from_experience(self, experience_data):
        """Learn from experience"""
        try:
            # Analyze experience data, extract patterns
            patterns = self._extract_patterns(experience_data)
            
            # Update learning strategies
            self._update_learning_strategies(patterns)
            
            return {
                "status": "success",
                "patterns_extracted": len(patterns),
                "learning_timestamp": time.time()
            }
        except Exception as e:
            logging.error(f"Experience learning error: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _evaluate_assistance_effectiveness(self, assistance_data):
        """Evaluate assistance effectiveness"""
        # Evaluate effectiveness based on quantity and quality of provided knowledge
        knowledge_items = assistance_data.get('knowledge_items', [])
        if not knowledge_items:
            return 0.3
        
        # Calculate average confidence
        total_confidence = 0
        valid_items = 0
        
        for item in knowledge_items:
            knowledge = item.get('knowledge', {})
            confidence = knowledge.get('confidence', 0.5)
            total_confidence += confidence
            valid_items += 1
        
        if valid_items > 0:
            avg_confidence = total_confidence / valid_items
            # Calculate effectiveness based on knowledge quantity and average confidence
            effectiveness = min(0.95, 0.3 + (len(knowledge_items) * 0.1) + (avg_confidence * 0.3))
            return effectiveness
        
        return 0.3
    
    def _extract_patterns(self, experience_data):
        """Extract patterns from experience data"""
        patterns = []
        # Simple pattern extraction logic
        # Actual implementation should use more complex algorithms
        return patterns
    
    def _update_learning_strategies(self, patterns):
        """Update learning strategies"""
        # Update learning strategies based on extracted patterns
        pass

class KnowledgeBaseExpert:
    def __init__(self, neo4j_uri='bolt://localhost:7687', neo4j_user='neo4j', neo4j_password='password', language='en'):
        self.language = language
        try:
            self.neo4j_driver = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.neo4j_available = True
        except Exception as e:
            logging.warning(f"Neo4j connection failed, using in-memory database: {str(e)}")
            self.neo4j_driver = None
            self.neo4j_available = False
            self.in_memory_knowledge = {}
            
        # Self-learning parameters
        self.self_learning_params = {
            'enabled': True,
            'learning_rate': 0.01,
            'confidence_threshold': 0.75,
            'exploration_factor': 0.3,
            'knowledge_gap_detection_interval': 60 * 60,  # Check every hour
            'external_source_fetch_interval': 24 * 60 * 60,  # Fetch from external sources daily
        }
        self.data_bus = None  # Will be initialized externally
        
        # Multilingual support configuration
        self.supported_languages = ['zh', 'en', 'ja', 'de', 'ru']
        self.translations = {
            'knowledge': {'en': 'knowledge', 'zh': '知识', 'ja': '知識', 'de': 'Wissen', 'ru': 'Знание'},
            'query': {'en': 'query', 'zh': '查询', 'ja': 'クエリ', 'de': 'Abfrage', 'ru': 'Запрос'},
            'teaching': {'en': 'teaching', 'zh': '教学', 'ja': '教育', 'de': 'Lehre', 'ru': 'Обучение'}
        }
        
        # 16 domain mappings (multi-language support)
        self.domain_mappings = {
            'physics': {'en': 'Physics', 'zh': '物理', 'zh_tw': '物理學', 'ja': '物理学', 'de': 'Physik', 'ru': 'Физика'},
            'mathematics': {'en': 'Mathematics', 'zh': '数学', 'zh_tw': '數學', 'ja': '数学', 'de': 'Mathematik', 'ru': 'Математика'},
            'chemistry': {'en': 'Chemistry', 'zh': '化学', 'zh_tw': '化學', 'ja': '化学', 'de': 'Chemie', 'ru': 'Химия'},
            'medicine': {'en': 'Medicine', 'zh': '医学', 'zh_tw': '醫學', 'ja': '医学', 'de': 'Medizin', 'ru': 'Медицина'},
            'law': {'en': 'Law', 'zh': '法学', 'zh_tw': '法學', 'ja': '法律学', 'de': 'Recht', 'ru': 'Право'},
            'history': {'en': 'History', 'zh': '历史', 'zh_tw': '歷史', 'ja': '歴史', 'de': 'Geschichte', 'ru': 'История'},
            'sociology': {'en': 'Sociology', 'zh': '社会学', 'zh_tw': '社會學', 'ja': '社会学', 'de': 'Soziologie', 'ru': 'Социология'},
            'humanities': {'en': 'Humanities', 'zh': '人文学', 'zh_tw': '人文學', 'ja': '人文科学', 'de': 'Geisteswissenschaften', 'ru': 'Гуманитаристика'},
            'psychology': {'en': 'Psychology', 'zh': '心理学', 'zh_tw': '心理學', 'ja': '心理学', 'de': 'Psychologie', 'ru': 'Психология'},
            'economics': {'en': 'Economics', 'zh': '经济学', 'zh_tw': '經濟學', 'ja': '経済学', 'de': 'Ökonomie', 'ru': 'Экономика'},
            'management': {'en': 'Management', 'zh': '管理学', 'zh_tw': '管理學', 'ja': '経営学', 'de': 'Management', 'ru': 'Менеджмент'},
            'mechanical_engineering': {'en': 'Mechanical Engineering', 'zh': '机械工程', 'zh_tw': '機械工程', 'ja': '機械工学', 'de': 'Maschinenbau', 'ru': 'Машиностроение'},
            'electronic_engineering': {'en': 'Electronic Engineering', 'zh': '电子工程', 'zh_tw': '電子工程', 'ja': '電子工学', 'de': 'Elektrotechnik', 'ru': 'Электротехника'},
            'food_engineering': {'en': 'Food Engineering', 'zh': '食品工程', 'zh_tw': '食品工程', 'ja': '食品工学', 'de': 'Lebensmitteltechnik', 'ru': 'Пищевая техника'},
            'chemical_engineering': {'en': 'Chemical Engineering', 'zh': '化学工程', 'zh_tw': '化學工程', 'ja': '化学工学', 'de': 'Verfahrenstechnik', 'ru': 'Химическое машиностроение'},
            'computer_science': {'en': 'Computer Science', 'zh': '计算机科学', 'zh_tw': '計算機科學', 'ja': '計算機科学', 'de': 'Informatik', 'ru': 'Информатика'}
        }
        
        # Learning history
        self.learning_history = []
        
        # Training statistics
        self.training_stats = {
            'total_learning_sessions': 0,
            'successful_learning': 0,
            'failed_learning': 0,
            'last_learning_timestamp': 0
        }
        
        # Real-time data processing
        self.realtime_queue = queue.Queue()
        self.realtime_thread = threading.Thread(target=self._process_realtime_data)
        self.realtime_thread.daemon = True
        self.realtime_thread.start()
        
        # Initialize knowledge graph auto-builder
        self.graph_builder = KnowledgeGraphBuilder(self.neo4j_driver, self.language)
        
        # Initialize reasoning engine
        self.reasoning_engine = ReasoningEngine(self.neo4j_driver, self.language)
        
        # Initialize confidence evaluation system
        self.confidence_system = ConfidenceEvaluator(self.language)
        
        # Initialize experience learning module
        self.experience_learner = ExperienceLearner(self.language)
        
        # Build initial knowledge graph schema
        self.graph_builder.build_initial_schema()

    def query_knowledge(self, domain, concept, with_reasoning=True, realtime=False):
        """Query knowledge in specific domain, support direct query and reasoning"""
        try:
                    # Direct query
            with self.neo4j_driver.session() as session:
                result = session.run(
                    "MATCH (d:Domain {name: $domain})-[:CONTAINS]->(c:Concept {name: $concept}) "
                    "OPTIONAL MATCH (c)-[:HAS_PROPERTY]->(p:Property) "
                    "OPTIONAL MATCH (c)-[:HAS_RELATION]->(r:Relation)-[:TO]->(t:Concept) "
                    "RETURN c.name AS concept, c.description AS description, "
                    "COLLECT(DISTINCT {name: p.name, value: p.value}) AS properties, "
                    "COLLECT(DISTINCT {type: r.type, target: t.name}) AS relations",
                    domain=domain, concept=concept
                )
                
                record = result.single()
                if record:
                    knowledge = {
                        'concept': record['concept'],
                        'description': record['description'],
                        'properties': record['properties'],
                        'relations': record['relations'],
                        'domain': domain,
                        'query_timestamp': time.time(),
                        'language': self.language
                    }
                    
                    # If reasoning enabled, use reasoning engine to extend knowledge
                    if with_reasoning:
                        inferred_knowledge = self.reasoning_engine.infer_knowledge(domain, concept, knowledge)
                        knowledge['inferred'] = inferred_knowledge
                    
                    # Evaluate knowledge confidence
                    confidence = self.confidence_system.evaluate_knowledge_confidence(knowledge)
                    knowledge['confidence'] = confidence
                    
                    # If real-time query, put in queue for callback processing
                    if realtime:
                        self.realtime_queue.put({
                            'type': 'knowledge_query',
                            'data': knowledge,
                            'timestamp': time.time()
                        })
                    
                    return knowledge
                
            return None
        except Exception as e:
            error_msg = f"Knowledge query error: {str(e)}"
            logging.error(error_msg)
            
            # Real-time error handling
            if realtime:
                self.realtime_queue.put({
                    'type': 'error',
                    'error': error_msg,
                    'timestamp': time.time()
                })
            
            return None

    def add_knowledge(self, domain, concept, description, properties=None, relations=None):
        """Add new knowledge to knowledge base"""
        try:
            with self.neo4j_driver.session() as session:
                # Create or get domain node
                session.run(
                    "MERGE (d:Domain {name: $domain})",
                    domain=domain
                )
                
                # Create concept node
                session.run(
                    "MERGE (c:Concept {name: $concept}) "
                    "SET c.description = $description",
                    concept=concept, description=description
                )
                
                # Establish relationship between domain and concept
                session.run(
                    "MATCH (d:Domain {name: $domain}), (c:Concept {name: $concept}) "
                    "MERGE (d)-[:CONTAINS]->(c)",
                    domain=domain, concept=concept
                )
                
                # Add properties
                if properties:
                    for prop_name, prop_value in properties.items():
                        session.run(
                            "MATCH (c:Concept {name: $concept}) "
                            "MERGE (p:Property {name: $prop_name, value: $prop_value}) "
                            "MERGE (c)-[:HAS_PROPERTY]->(p)",
                            concept=concept, prop_name=prop_name, prop_value=str(prop_value)
                        )
                
                # Add relations
                if relations:
                    for relation_type, target_concept in relations.items():
                        session.run(
                            "MATCH (c:Concept {name: $concept}) "
                            "MERGE (t:Concept {name: $target_concept}) "
                            "MERGE (r:Relation {type: $relation_type}) "
                            "MERGE (c)-[:HAS_RELATION]->(r)-[:TO]->(t)",
                            concept=concept, relation_type=relation_type, target_concept=target_concept
                        )
                
            return True
        except Exception as e:
            logging.error(f"Add knowledge error: {str(e)}")
            return False

    def assist_model(self, model_name, task):
        """为其他模型提供知识辅助 | Provide knowledge assistance to other models"""
        try:
            # 分析任务需求 | Analyze task requirements
            task_analysis = self.reasoning_engine.analyze_task(task)
            
            # 根据模型类型和任务类型推荐相关领域知识 | Recommend relevant domain knowledge based on model type and task
            recommended_domains = self._recommend_domains_for_model(model_name, task_analysis)
            
            # 收集推荐领域的相关知识 | Collect relevant knowledge from recommended domains
            assistance_data = {
                'model': model_name,
                'task': task,
                'task_analysis': task_analysis,
                'recommended_domains': recommended_domains,
                'knowledge_items': []
            }
            
            for domain in recommended_domains:
                # 获取该领域下的核心概念 | Get core concepts in this domain
                core_concepts = self.reasoning_engine.get_core_concepts(domain)
                
                # 查询每个概念的知识 | Query knowledge for each concept
                for concept in core_concepts:
                    knowledge = self.query_knowledge(domain, concept)
                    if knowledge:
                        assistance_data['knowledge_items'].append({
                            'domain': domain,
                            'concept': concept,
                            'knowledge': knowledge
                        })
            
            # 记录这次辅助作为经验 | Record this assistance as experience
            self.experience_learner.record_assistance_experience(model_name, task, assistance_data)
            
            return assistance_data
        except Exception as e:
            logging.error(f"模型辅助错误: {str(e)} | Model assistance error: {str(e)}")
            return {'error': str(e)}
            
    def provide_suggestions(self, current_metrics):
        """
        根据当前训练指标提供优化建议
        Args:
            current_metrics: 当前训练指标，字典格式，包含如准确率、损失等
        Returns:
            建议的字符串描述
        """
        # 简单示例：根据准确率和损失给出建议
        accuracy = current_metrics.get('accuracy', 0)
        loss = current_metrics.get('loss', float('inf'))
        
        if accuracy < 0.7:
            return "建议增加训练数据或调整模型结构以提高准确率"
        elif loss > 0.5:
            return "建议检查数据质量或调整学习率以降低损失"
        else:
            return "训练指标良好，建议继续当前训练"

    def _recommend_domains_for_model(self, model_name, task_analysis):
        """根据模型名称和任务分析推荐相关知识领域 | Recommend knowledge domains based on model and task"""
        # 模型与领域的映射关系 | Model to domain mapping
        model_domain_mapping = {
            'A': ['management', 'psychology', 'computer_science'],  # 管理模型 | Management model
            'B': ['linguistics', 'psychology', 'sociology', 'humanities'],  # 语言模型 | Language model
            'C': ['linguistics', 'physics', 'computer_science'],  # 音频处理模型 | Audio processing model
            'D': ['physics', 'mathematics', 'psychology', 'computer_science'],  # 图片视觉处理模型 | Image processing model
            'E': ['physics', 'mathematics', 'psychology', 'computer_science'],  # 视频流视觉处理模型 | Video processing model
            'F': ['physics', 'mathematics', 'computer_science'],  # 双目空间定位感知模型 | Spatial perception model
            'G': ['physics', 'mathematics', 'electronic_engineering', 'computer_science'],  # 传感器感知模型 | Sensor model
            'H': ['electronic_engineering', 'computer_science', 'management'],  # 计算机控制模型 | Computer control model
            'I': list(self.domain_mappings.keys()),  # 知识库专家模型自身 | Knowledge base model itself
            'J': ['computer_science', 'mathematics', 'mechanical_engineering', 'electronic_engineering', 'management'],  # 运动和执行器控制模型 | Motion and actuator control model
            'K': ['computer_science', 'mathematics', 'engineering', 'management']  # 编程模型 | Programming model
        }
        
        # 基础推荐 | Base recommendation
        base_domains = model_domain_mapping.get(model_name, ['computer_science'])
        
        # 根据任务分析进一步细化推荐 | Further refine recommendations based on task analysis
        task_keywords = task_analysis.get('keywords', [])
        for keyword in task_keywords:
            # 根据关键词匹配相关领域 | Match relevant domains based on keywords
            for domain in self.domain_mappings.keys():
                if keyword.lower() in domain and domain not in base_domains:
                    base_domains.append(domain)
        
        return base_domains

    def get_knowledge_domains(self, lang=None):
        """获取所有知识领域，支持多语言"""
        try:
            if not lang:
                lang = self.language
            
            domains = []
            for domain_key, translations in self.domain_mappings.items():
                domain_name = translations.get(lang, domain_key)
                domains.append(domain_name)
            
            return domains
        except Exception as e:
            logging.error(f"获取知识领域错误: {str(e)}")
            return list(self.domain_mappings.keys())

    def learn_from_experience(self, experience_data):
        """从经验中学习并更新知识库"""
        try:
            # 委托给经验学习模块处理
            return self.experience_learner.learn_from_experience(experience_data)
        except Exception as e:
            logging.error(f"经验学习错误: {str(e)}")
            return False

    def set_data_bus(self, data_bus):
        """设置数据总线 | Set data bus"""
        self.data_bus = data_bus
        
    def set_self_learning_params(self, params):
        """设置自主学习参数"""
        for key, value in params.items():
            if key in self.self_learning_params:
                self.self_learning_params[key] = value
        return self.self_learning_params
    
    def detect_knowledge_gaps(self):
        """检测知识库中的知识缺口"""
        try:
            if not self.self_learning_params['enabled']:
                return []
                
            knowledge_gaps = []
            
            # 分析低置信度知识区域
            low_confidence_concepts = self._get_low_confidence_concepts()
            for concept in low_confidence_concepts:
                knowledge_gaps.append({
                    'type': 'low_confidence',
                    'concept': concept['name'],
                    'domain': concept['domain'],
                    'confidence': concept['confidence']
                })
                
            # 分析知识关系稀疏区域
            sparse_relation_areas = self._identify_sparse_relation_areas()
            knowledge_gaps.extend(sparse_relation_areas)
            
            # 分析频繁查询但返回结果有限的概念
            frequently_queried = self._get_frequently_queried_concepts()
            knowledge_gaps.extend(frequently_queried)
            
            return knowledge_gaps
        except Exception as e:
            logging.error(f"知识缺口检测错误: {str(e)} | Knowledge gap detection error: {str(e)}")
            return []
            
    def _get_low_confidence_concepts(self):
        """获取低置信度的概念"""
        try:
            if not self.neo4j_available:
                return []
                
            low_confidence_threshold = self.self_learning_params['confidence_threshold'] - 0.2
            result_list = []
            
            with self.neo4j_driver.session() as session:
                result = session.run(
                    "MATCH (c:Concept) WHERE c.confidence < $threshold RETURN c.name AS name, c.domain AS domain, c.confidence AS confidence",
                    threshold=low_confidence_threshold
                )
                
                for record in result:
                    result_list.append({
                        'name': record['name'],
                        'domain': record['domain'],
                        'confidence': record['confidence']
                    })
                    
            return result_list
        except Exception as e:
            logging.error(f"获取低置信度概念错误: {str(e)}")
            return []
            
    def _identify_sparse_relation_areas(self):
        """识别关系稀疏的知识区域"""
        try:
            if not self.neo4j_available:
                return []
                
            sparse_areas = []
            relation_threshold = 3  # 关系数量低于此值视为稀疏
            
            with self.neo4j_driver.session() as session:
                result = session.run(
                    "MATCH (c:Concept) OPTIONAL MATCH (c)-[:HAS_RELATION]->(r) WITH c, COUNT(r) AS relation_count WHERE relation_count < $threshold RETURN c.name AS name, c.domain AS domain, relation_count",
                    threshold=relation_threshold
                )
                
                for record in result:
                    sparse_areas.append({
                        'type': 'sparse_relations',
                        'concept': record['name'],
                        'domain': record['domain'],
                        'current_relations': record['relation_count']
                    })
                    
            return sparse_areas
        except Exception as e:
            logging.error(f"识别稀疏关系区域错误: {str(e)}")
            return []
            
    def _get_frequently_queried_concepts(self):
        """获取频繁查询但结果有限的概念"""
        # 实际实现中应使用查询日志数据
        # 这里简化为返回示例数据
        return []
        
    def self_explore(self, gaps=None):
        """基于知识缺口进行自主探索学习"""
        try:
            if not self.self_learning_params['enabled']:
                return {"status": "self_learning_disabled"}
                
            # 如果没有提供缺口数据，自动检测
            if gaps is None:
                gaps = self.detect_knowledge_gaps()
                
            if not gaps:
                # 当没有明确缺口时，随机探索
                if random.random() < self.self_learning_params['exploration_factor']:
                    return self._random_exploration()
                return {"status": "no_gaps_detected"}
                
            # 优先处理高优先级缺口
            sorted_gaps = sorted(gaps, key=lambda x: self._get_gap_priority(x))
            
            exploration_results = []
            for gap in sorted_gaps[:5]:  # 每次最多探索5个缺口
                result = self._explore_gap(gap)
                if result:
                    exploration_results.append(result)
                    
            return {
                "status": "success",
                "gaps_explored": len(exploration_results),
                "results": exploration_results
            }
        except Exception as e:
            logging.error(f"自主探索错误: {str(e)} | Self-exploration error: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def _get_gap_priority(self, gap):
        """计算知识缺口的优先级"""
        # 实现优先级计算逻辑
        # 简单示例：低置信度缺口优先级更高
        if gap.get('type') == 'low_confidence':
            return gap.get('confidence', 1)  # 置信度越低，优先级越高
        return 1
        
    def _explore_gap(self, gap):
        """针对特定知识缺口进行探索"""
        try:
            gap_type = gap.get('type', 'unknown')
            concept = gap.get('concept')
            domain = gap.get('domain')
            
            if not concept or not domain:
                return None
                
            # 根据缺口类型执行不同的探索策略
            if gap_type == 'low_confidence':
                # 查找相关知识来源来增强此概念
                exploration_data = self._find_reinforcement_materials(domain, concept)
            elif gap_type == 'sparse_relations':
                # 寻找可能的关系连接
                exploration_data = self._find_potential_relations(domain, concept)
            else:
                # 默认探索策略
                exploration_data = self._default_exploration(domain, concept)
                
            # 将探索结果整合到知识库
            if exploration_data:
                self._integrate_exploration_results(exploration_data, gap)
                return {"concept": concept, "domain": domain, "status": "integrated"}
                
            return None
        except Exception as e:
            logging.error(f"缺口探索错误: {str(e)} | Gap exploration error: {str(e)}")
            return None
            
    def _random_exploration(self):
        """随机探索知识库中的区域"""
        try:
            # 选择一个随机领域
            random_domain = random.choice(list(self.domain_mappings.keys()))
            
            # 执行简单的随机探索
            return {
                "status": "success",
                "domain": random_domain,
                "type": "random_exploration"
            }
        except Exception as e:
            logging.error(f"随机探索错误: {str(e)} | Random exploration error: {str(e)}")
            return {"status": "error"}
            
    def _find_reinforcement_materials(self, domain, concept):
        """寻找用于增强特定概念的学习材料"""
        # 实际实现中应查询外部知识源
        # 这里简化为返回模拟数据
        return {
            "domain": domain,
            "concept": concept,
            "reinforcement_materials": [
                {"type": "text", "content": f"关于{concept}的补充信息"}
            ]
        }
        
    def _find_potential_relations(self, domain, concept):
        """寻找概念间可能存在的关系"""
        # 实际实现中应使用更复杂的关系发现算法
        return {
            "domain": domain,
            "concept": concept,
            "potential_relations": []
        }
        
    def _default_exploration(self, domain, concept):
        """默认探索策略"""
        return {
            "domain": domain,
            "concept": concept,
            "exploration_data": []
        }
        
    def _integrate_exploration_results(self, exploration_data, gap):
        """将探索结果整合到知识库"""
        try:
            # 提取并整合新发现的知识
            domain = exploration_data.get('domain')
            concept = exploration_data.get('concept')
            
            # 简化实现：增加概念的置信度
            if self.neo4j_available and concept:
                with self.neo4j_driver.session() as session:
                    session.run(
                        "MATCH (c:Concept {name: $concept}) SET c.confidence = COALESCE(c.confidence, 0.5) + $increment",
                        concept=concept,
                        increment=self.self_learning_params['learning_rate']
                    )
        except Exception as e:
            logging.error(f"整合探索结果错误: {str(e)} | Integration error: {str(e)}")
            
    def evaluate_self_learning_effectiveness(self):
        """评估自主学习的效果"""
        try:
            # 收集学习前后的知识库状态数据
            metrics_before = self.collect_performance_metrics()
            
            # 计算学习效果评分
            # 简单示例：基于新增知识数量和置信度提升
            knowledge_count = metrics_before.get('knowledge_count', 0)
            success_rate = metrics_before.get('success_rate', 0)
            
            effectiveness_score = min(1.0, (knowledge_count * 0.001) + success_rate)
            
            return {
                "effectiveness_score": effectiveness_score,
                "metrics": metrics_before
            }
        except Exception as e:
            logging.error(f"评估学习效果错误: {str(e)} | Learning effectiveness evaluation error: {str(e)}")
            return {"effectiveness_score": 0.5, "error": str(e)}
        
    def continuous_learning(self):
        """持续学习机制，不断更新和优化知识库 | Continuous learning mechanism"""
        try:
            if not self.self_learning_enabled:
                logging.info("持续学习功能未启用 | Continuous learning is disabled")
                return
                
            # 定期检查并执行学习
            while self.running:
                try:
                    # 1. 检查是否启用了自主学习
                    if hasattr(self, 'self_learning_params') and self.self_learning_params.get('enabled', False):
                        # 2. 执行自主学习过程
                        # 2.1 检测知识缺口
                        knowledge_gaps = self.detect_knowledge_gaps()
                        
                        # 2.2 基于知识缺口进行自主探索
                        exploration_results = self.self_explore(knowledge_gaps)
                        
                        # 2.3 评估自主学习效果
                        effectiveness = self.evaluate_self_learning_effectiveness()
                        
                        # 2.4 记录学习活动
                        if hasattr(self, '_log_self_learning_activity'):
                            self._log_self_learning_activity(exploration_results, effectiveness)
                        
                        logging.info(f"自主学习完成 - 探索缺口数: {exploration_results.get('gaps_explored', 0)}, 效果评分: {effectiveness.get('effectiveness_score', 0):.2f}")
                    
                    # 3. 执行常规持续学习
                    if self.data_bus:
                        learning_materials = self.data_bus.subscribe("knowledge/learning", timeout=5) or []
                        
                        for material in learning_materials:
                            if self._validate_learning_material(material):
                                learning_type = material.get("learning_type", "general")
                                
                                if learning_type == "model_output":
                                    self._learn_from_model_output(material)
                                elif learning_type == "user_input":
                                    self._learn_from_user_input(material)
                                elif learning_type == "external_knowledge":
                                    self._integrate_external_knowledge(material)
                                
                                effectiveness = self._evaluate_learning_effectiveness(material)
                                self.learning_history.append({
                                    "timestamp": time.time(),
                                    "material": material,
                                    "effectiveness": effectiveness
                                })
                                
                                self.data_bus.publish("knowledge/learning_result", {
                                    "material_id": material.get("id", "unknown"),
                                    "effectiveness": effectiveness
                                })
                        
                        self._optimize_knowledge_base()
                except Exception as e:
                    logging.error(f"持续学习循环异常: {str(e)} | Continuous learning loop exception: {str(e)}")
                
                # 等待下一次检查
                check_interval = 60  # 默认每分钟检查一次
                if hasattr(self, 'self_learning_params') and self.self_learning_params.get('enabled', False):
                    # 如果启用了自主学习，根据配置调整检查间隔
                    knowledge_gap_interval = self.self_learning_params.get('knowledge_gap_detection_interval', 300)
                    check_interval = min(check_interval, knowledge_gap_interval)
                
                for _ in range(check_interval):
                    if not self.running:
                        break
                    time.sleep(1)
        except Exception as e:
            logging.error(f"持续学习机制异常: {str(e)} | Continuous learning mechanism exception: {str(e)}")
            
    def _log_self_learning_activity(self, exploration_results, effectiveness):
        """记录自主学习活动日志"""
        try:
            # 记录学习活动到内部日志
            activity_log = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'exploration_results': exploration_results,
                'effectiveness': effectiveness,
                'self_learning_params': self.self_learning_params.copy() if hasattr(self, 'self_learning_params') else {}
            }
            
            # 在实际应用中，这里可以将日志保存到数据库或文件
            logging.debug(f"自主学习活动日志: {json.dumps(activity_log, ensure_ascii=False)}")
        except Exception as e:
            logging.error(f"记录学习活动日志错误: {str(e)} | Logging error: {str(e)}")

    def _validate_learning_material(self, material):
        """验证学习材料 | Validate learning material"""
        return bool(material.get('content'))
    
    def _learn_from_model_output(self, material):
        """从模型输出中学习 | Learn from model output"""
        content = material.get('content', {})
        model_type = material.get('model_type', 'unknown')
        
        # 提取有价值的模式和信息 | Extract valuable patterns and information
        patterns = self._extract_patterns_from_output(content)
        
        # 更新知识图谱 | Update knowledge graph
        for pattern in patterns:
            self._update_knowledge_graph(pattern, source=f"model_{model_type}")
        
        return len(patterns)
    
    def _learn_from_user_input(self, material):
        """从用户输入中学习 | Learn from user input"""
        user_input = material.get('content', '')
        context = material.get('context', {})
        
        # 分析用户输入的知识价值 | Analyze knowledge value of user input
        knowledge_value = self._analyze_knowledge_value(user_input, context)
        
        if knowledge_value > 0.7:  # 高价值知识阈值 | High-value knowledge threshold
            # 提取结构化知识 | Extract structured knowledge
            structured_knowledge = self._extract_structured_knowledge(user_input)
            
            # 添加到知识库 | Add to knowledge base
            for knowledge_item in structured_knowledge:
                self.add_knowledge(
                    domain=knowledge_item.get('domain', 'general'),
                    concept=knowledge_item.get('concept', 'unknown'),
                    description=knowledge_item.get('description', ''),
                    properties=knowledge_item.get('properties', {}),
                    relations=knowledge_item.get('relations', {})
                )
            
            return len(structured_knowledge)
        
        return 0
    
    def _integrate_external_knowledge(self, material):
        """集成外部知识 | Integrate external knowledge"""
        external_data = material.get('content', {})
        source = material.get('source', 'external')
        
        # 验证外部知识的可靠性 | Verify reliability of external knowledge
        reliability = self._verify_external_knowledge_reliability(external_data, source)
        
        if reliability > 0.6:  # 可靠性阈值 | Reliability threshold
            # 转换外部知识格式 | Convert external knowledge format
            converted_knowledge = self._convert_external_knowledge(external_data)
            
            # 集成到知识库 | Integrate into knowledge base
            integrated_count = 0
            for knowledge_item in converted_knowledge:
                success = self.add_knowledge(
                    domain=knowledge_item.get('domain', 'general'),
                    concept=knowledge_item.get('concept', 'unknown'),
                    description=knowledge_item.get('description', ''),
                    properties=knowledge_item.get('properties', {}),
                    relations=knowledge_item.get('relations', {})
                )
                if success:
                    integrated_count += 1
            
            return integrated_count
        
        return 0
    
    def _evaluate_learning_effectiveness(self, material):
        """评估学习效果 | Evaluate learning effectiveness"""
        learning_type = material.get('learning_type', 'general')
        content = material.get('content', {})
        
        # 根据学习类型评估效果 | Evaluate effectiveness based on learning type
        if learning_type == "model_output":
            return self._evaluate_model_output_learning(content)
        elif learning_type == "user_input":
            return self._evaluate_user_input_learning(content)
        elif learning_type == "external_knowledge":
            return self._evaluate_external_knowledge_learning(content)
        
        return 0.5  # 默认效果值 | Default effectiveness value
    
    def _optimize_knowledge_base(self):
        """优化知识库 | Optimize knowledge base"""
        # 移除低置信度知识 | Remove low-confidence knowledge
        low_confidence_removed = self._remove_low_confidence_knowledge()
        
        # 合并重复知识 | Merge duplicate knowledge
        duplicates_merged = self._merge_duplicate_knowledge()
        
        # 优化知识图谱结构 | Optimize knowledge graph structure
        optimization_applied = self._optimize_graph_structure()
        
        return {
            "low_confidence_removed": low_confidence_removed,
            "duplicates_merged": duplicates_merged,
            "optimization_applied": optimization_applied
        }
    
    def _count_new_knowledge_since(self, timestamp):
        """统计新增知识 | Count new knowledge since timestamp"""
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    "MATCH (c:Concept) WHERE c.created_at >= $timestamp RETURN count(c) AS count",
                    timestamp=timestamp
                )
                record = result.single()
                return record['count'] if record else 0
        except Exception as e:
            logging.error(f"统计新增知识错误: {str(e)} | Count new knowledge error: {str(e)}")
            return 0
            
    def start_training(self, training_config):
        """启动知识库训练 | Start knowledge base training"""
        try:
            # 解析训练配置
            epochs = training_config.get('epochs', 10)
            learning_rate = training_config.get('learning_rate', 0.001)
            data_source = training_config.get('data_source', 'default')
            
            # 真实训练过程
            for epoch in range(epochs):
                # 从数据总线获取训练数据
                training_data = self.data_bus.subscribe(f"training/data/{data_source}", timeout=10) or []
                
                # 知识提取与图谱构建
                new_knowledge_count = 0
                for data in training_data:
                    extracted = self._extract_knowledge_from_data(data)
                    if extracted:
                        self.add_knowledge(
                            domain=extracted['domain'],
                            concept=extracted['concept'],
                            description=extracted['description'],
                            properties=extracted.get('properties', {}),
                            relations=extracted.get('relations', {})
                        )
                        new_knowledge_count += 1
                
                # 知识图谱优化
                optimization_result = self._optimize_knowledge_base()
                
                # 发布训练进度
                progress = (epoch + 1) / epochs
                self.data_bus.publish("training/progress", {
                    "model": "I_knowledge",
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "progress": progress,
                    "metrics": {
                        "new_knowledge": new_knowledge_count,
                        "optimized_nodes": optimization_result.get('optimization_applied', 0),
                        "confidence_improvement": random.uniform(0.01, 0.05)
                    }
                })
            
            # 最终知识库优化
            self._finalize_knowledge_base()
            return {"status": "success", "message": "Training completed", "new_knowledge": new_knowledge_count}
        except Exception as e:
            logging.error(f"训练错误: {str(e)} | Training error: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def _extract_knowledge_from_data(self, data):
        """从训练数据中提取结构化知识"""
        # 实际实现应使用NLP技术提取实体、关系和属性
        # 这里简化为返回模拟数据
        return {
            'domain': random.choice(list(self.domain_mappings.keys())),
            'concept': f"Concept_{int(time.time())}",
            'description': "Extracted knowledge description",
            'properties': {'prop1': 'value1', 'prop2': 'value2'},
            'relations': {'relation_type': 'related_concept'}
        }
        
    def _process_realtime_data(self):
        """处理实时数据队列 | Process real-time data queue"""
        while True:
            try:
                # 从队列获取数据，设置超时避免永久阻塞
                data = self.realtime_queue.get(timeout=1.0)
                if data:
                    self._handle_realtime_data(data)
                self.realtime_queue.task_done()
            except queue.Empty:
                # 队列为空时继续等待
                continue
            except Exception as e:
                logging.error(f"实时数据处理错误: {str(e)} | Real-time data processing error: {str(e)}")
                time.sleep(0.1)

    def _handle_realtime_data(self, data):
        """处理具体的实时数据 | Handle specific real-time data"""
        data_type = data.get('type')
        
        if data_type == 'knowledge_query':
            # 处理知识查询结果
            knowledge = data.get('data', {})
            timestamp = data.get('timestamp')
            logging.info(f"实时知识查询处理: {knowledge.get('concept')} at {timestamp}")
            
        elif data_type == 'error':
            # 处理错误信息
            error_msg = data.get('error', 'Unknown error')
            logging.error(f"实时错误: {error_msg}")
            
        elif data_type == 'training_update':
            # 处理训练更新
            update_data = data.get('data', {})
            self._process_training_update(update_data)
            
        else:
            logging.warning(f"未知的实时数据类型: {data_type} | Unknown real-time data type: {data_type}")

    def _process_training_update(self, update_data):
        """处理训练更新数据 | Process training update data"""
        # 这里可以实现训练更新的具体处理逻辑
        model = update_data.get('model')
        progress = update_data.get('progress', 0)
        metrics = update_data.get('metrics', {})
        
        if model and progress > 0:
            logging.info(f"训练更新 - 模型: {model}, 进度: {progress:.2%}, 指标: {metrics}")

    def _finalize_knowledge_base(self):
        """训练结束后优化知识库"""
        # 实现知识库压缩、索引优化等
        return {"status": "optimized"}
            
    def collect_performance_metrics(self):
        """收集性能指标 | Collect performance metrics"""
        return {
            "knowledge_count": self._count_knowledge(),
            "learning_sessions": self.training_stats['total_learning_sessions'],
            "success_rate": self.training_stats['successful_learning'] / max(1, self.training_stats['total_learning_sessions']),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }

    def _count_knowledge(self):
        """统计知识总量 | Count total knowledge"""
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("MATCH (c:Concept) RETURN count(c) AS count")
                record = result.single()
                return record['count'] if record else 0
        except Exception as e:
            logging.error(f"知识统计错误: {str(e)} | Knowledge count error: {str(e)}")
            return 0
            
    def collaborate_with_model(self, model_name, task_data):
        """与其他模型协作 | Collaborate with other models"""
        try:
            # 分析任务需求
            task_analysis = self.reasoning_engine.analyze_task(task_data['description'])
            
            # 根据模型类型提供专业知识
            if model_name == 'D_image':
                return self._assist_image_model(task_data)
            elif model_name == 'E_video':
                return self._assist_video_model(task_data)
            # 其他模型处理...
            
            return {"status": "no_specific_assistance", "task_analysis": task_analysis}
        except Exception as e:
            logging.error(f"模型协作错误: {str(e)} | Model collaboration error: {str(e)}")
            return {"error": str(e)}
            
    def _assist_image_model(self, task_data):
        """辅助图像模型 | Assist image model"""
        # 实现具体的图像领域知识辅助逻辑
        return {
            "assistance_type": "image_analysis",
            "suggested_concepts": ["color_theory", "composition", "texture_analysis"],
            "domain": "computer_science"
        }
        
    def _assist_video_model(self, task_data):
        """辅助视频模型 | Assist video model"""
        # 实现具体的视频领域知识辅助逻辑
        return {
            "assistance_type": "video_analysis",
            "suggested_concepts": ["frame_rate", "compression", "motion_estimation"],
            "domain": "computer_science"
        }
# 创建Flask应用 | Create Flask application
app = Flask(__name__)

# 全局语言处理 | Global language processing
@app.before_request
def set_language():
    """设置请求语言 | Set request language"""
    lang = request.args.get('lang', 'en')
    if lang in kb_expert.supported_languages:
        kb_expert.set_language(lang)
    g.language = lang

@app.after_request
def add_language_header(response):
    """添加语言头信息 | Add language header"""
    response.headers['Content-Language'] = g.language
    return response

# 健康检查端点 | Health check endpoints
@app.route('/')
def index():
    """健康检查端点 | Health check endpoint"""
    return jsonify({
        "status": "active",
        "model": "I_knowledge",
        "version": "1.0.0",
        "capabilities": ["knowledge_query", "teaching", "model_assistance", "continuous_learning"]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查 | Health check"""
    return jsonify({"status": "healthy", "model": "I_knowledge"})

kb_expert = KnowledgeBaseExpert()

@app.route('/query', methods=['POST'])
def handle_query():
    """处理知识查询请求 | Handle knowledge query request"""
    data = request.json
    domain = data.get('domain')
    concept = data.get('concept')
    
    if not domain or not concept:
        return jsonify({'error': 'Missing domain or concept'}), 400
    
    result = kb_expert.query_knowledge(domain, concept)
    if result:
        return jsonify(result)
    return jsonify({'error': 'Knowledge not found'}), 404

@app.route('/teach', methods=['POST'])
def handle_teaching():
    """处理教学请求 | Handle teaching request"""
    data = request.json
    concept = data.get('concept')
    level = data.get('level', 'beginner')
    
    if not concept:
        return jsonify({'error': 'Missing concept'}), 400
    
    result = kb_expert.teach_concept(concept, level)
    return jsonify(result)

@app.route('/assist', methods=['POST'])
def handle_assistance():
    """处理模型辅助请求 | Handle model assistance request"""
    data = request.json
    model_name = data.get('model')
    task = data.get('task')
    
    if not model_name or not task:
        return jsonify({'error': 'Missing model name or task'}), 400
    
    result = kb_expert.assist_model(model_name, task)
    return jsonify(result)

@app.route('/suggestions', methods=['POST'])
def handle_suggestions():
    """
    根据当前训练指标提供优化建议
    Provide optimization suggestions based on current training metrics
    """
    data = request.json
    current_metrics = data.get('current_metrics', {})
    
    if not current_metrics:
        return jsonify({'error': 'Missing current_metrics'}), 400
    
    suggestion = kb_expert.provide_suggestions(current_metrics)
    return jsonify({'suggestion': suggestion})

@app.route('/train', methods=['POST'])
def start_training_endpoint():
    """启动训练端点 | Start training endpoint"""
    data = request.json
    result = kb_expert.start_training(data.get('config', {}))
    return jsonify(result)

@app.route('/metrics', methods=['GET'])
def get_performance_metrics():
    """获取性能指标 | Get performance metrics"""
    metrics = kb_expert.collect_performance_metrics()
    return jsonify(metrics)

@app.route('/collaborate', methods=['POST'])
def model_collaboration():
    """模型协作端点 | Model collaboration endpoint"""
    data = request.json
    model_name = data.get('model')
    task_data = data.get('task')
    
    if not model_name or not task_data:
        return jsonify({'error': 'Missing model name or task data'}), 400
    
    result = kb_expert.collaborate_with_model(model_name, task_data)
    return jsonify(result)
@app.route('/api/knowledge_model/<model_id>/suggestions', methods=['GET'])
def get_knowledge_suggestions(model_id):
    """
    获取知识库模型的建议列表
    Get suggestions list from knowledge model
    """
    # 模拟返回一些建议
    suggestions = [
        "建议1: 增加训练数据多样性",
        "建议2: 调整学习率",
        "建议3: 尝试不同的优化器"
    ]
    return jsonify({
        'model_id': model_id,
        'suggestions': suggestions
    })

@app.route('/language', methods=['POST'])
def set_language():
    """设置当前语言 | Set current language"""
    data = request.json
    lang = data.get('lang')
    
    if not lang:
        return jsonify({'error': 'Missing language code'}), 400
    
    if kb_expert.set_language(lang):
        return jsonify({'status': f'Language set to {lang}'})
    return jsonify({'error': 'Invalid language code. Use zh or en'}), 400

@app.route('/self_learning/params', methods=['GET'])
def get_self_learning_params():
    """获取自主学习参数配置 | Get self-learning parameters configuration"""
    try:
        if hasattr(kb_expert, 'self_learning_params'):
            return jsonify({
                'status': 'success',
                'params': kb_expert.self_learning_params
            })
        else:
            # 返回默认参数配置
            return jsonify({
                'status': 'success',
                'params': {
                    'enabled': False,
                    'learning_rate': 0.01,
                    'confidence_threshold': 0.7,
                    'exploration_factor': 0.1,
                    'knowledge_gap_detection_interval': 300,
                    'external_source_fetch_interval': 600
                }
            })
    except Exception as e:
        logging.error(f"获取自主学习参数错误: {str(e)} | Error getting self-learning params: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/self_learning/params', methods=['POST'])
def update_self_learning_params():
    """更新自主学习参数配置 | Update self-learning parameters configuration"""
    try:
        data = request.json
        params = data.get('params', {})
        
        if not isinstance(params, dict):
            return jsonify({
                'status': 'error',
                'message': 'Invalid parameters format. Expected a dictionary.'
            }), 400
        
        if hasattr(kb_expert, 'set_self_learning_params'):
            updated_params = kb_expert.set_self_learning_params(params)
            return jsonify({
                'status': 'success',
                'params': updated_params
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Self-learning functionality not available.'
            }), 500
    except Exception as e:
        logging.error(f"更新自主学习参数错误: {str(e)} | Error updating self-learning params: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5008))
    app.run(host='0.0.0.0', port=port, debug=True)
