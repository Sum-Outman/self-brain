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

import os
import json
import logging
import re
import math
from datetime import datetime
import hashlib
import pickle
from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SearchEngine")

class KnowledgeSearchEngine:
    """知识库搜索引擎，用于高效搜索知识库内容"""
    
    def __init__(self, storage_path, metadata_manager=None):
        """初始化知识库搜索引擎
        
        Args:
            storage_path: 存储路径
            metadata_manager: 元数据管理器实例
        """
        self.storage_path = storage_path
        self.metadata_manager = metadata_manager
        self.index_file = os.path.join(storage_path, 'search_index.pkl')
        
        # 倒排索引结构
        self.inverted_index = defaultdict(list)  # 词项 -> 文档ID列表
        self.document_terms = defaultdict(Counter)  # 文档ID -> 词频统计
        self.idf = {}  # 逆文档频率
        self.total_docs = 0  # 文档总数
        
        # 确保存储路径存在
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            logger.info(f"搜索索引存储路径已创建或已存在: {self.storage_path}")
        except Exception as e:
            logger.error(f"创建搜索索引存储路径失败: {str(e)}")
            # 创建临时目录作为回退
            self.storage_path = os.path.join(os.environ.get('TEMP', '/tmp'), 'self_brain_search_temp')
            os.makedirs(self.storage_path, exist_ok=True)
            self.index_file = os.path.join(self.storage_path, 'search_index.pkl')
            logger.warning(f"使用临时存储路径作为回退: {self.storage_path}")
        
        # 加载现有索引
        self.load_index()
    
    def tokenize(self, text: str) -> List[str]:
        """将文本分词
        
        Args:
            text: 输入文本
        
        Returns:
            词项列表
        """
        try:
            # 转小写
            text = text.lower()
            # 移除非字母数字字符，保留空格
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            # 分词并移除空字符串
            tokens = [token for token in text.split() if token]
            return tokens
        except Exception as e:
            logger.error(f"分词失败: {str(e)}")
            return []
    
    def add_document(self, doc_id: str, title: str, content: str):
        """添加文档到索引
        
        Args:
            doc_id: 文档ID
            title: 文档标题
            content: 文档内容
        """
        try:
            # 对标题和内容进行分词
            title_tokens = self.tokenize(title)
            content_tokens = self.tokenize(content)
            
            # 合并标题和内容的词项，标题词项权重更高
            all_tokens = title_tokens * 2 + content_tokens  # 标题权重是内容的2倍
            
            # 统计词频
            term_counts = Counter(all_tokens)
            
            # 更新倒排索引和文档词频
            for term, count in term_counts.items():
                # 如果文档还没有在这个词项的列表中，添加它
                if not any(doc_info[0] == doc_id for doc_info in self.inverted_index[term]):
                    self.inverted_index[term].append([doc_id, 0])  # 初始TF为0
                
                # 更新文档词频
                self.document_terms[doc_id] = term_counts
            
            # 更新文档总数
            self.total_docs = len(self.document_terms)
            
            # 重新计算IDF
            self._calculate_idf()
            
            logger.info(f"成功添加文档到索引: {doc_id}")
        except Exception as e:
            logger.error(f"添加文档到索引失败: {str(e)}")
    
    def remove_document(self, doc_id: str):
        """从索引中移除文档
        
        Args:
            doc_id: 文档ID
        """
        try:
            # 如果文档不存在，直接返回
            if doc_id not in self.document_terms:
                logger.warning(f"文档不存在于索引中: {doc_id}")
                return
            
            # 从倒排索引中移除该文档
            for term in self.document_terms[doc_id]:
                self.inverted_index[term] = [doc_info for doc_info in self.inverted_index[term] if doc_info[0] != doc_id]
                
                # 如果该词项没有文档了，从索引中删除
                if not self.inverted_index[term]:
                    del self.inverted_index[term]
            
            # 从文档词频中移除
            del self.document_terms[doc_id]
            
            # 更新文档总数
            self.total_docs = len(self.document_terms)
            
            # 重新计算IDF
            self._calculate_idf()
            
            logger.info(f"成功从索引中移除文档: {doc_id}")
        except Exception as e:
            logger.error(f"从索引中移除文档失败: {str(e)}")
    
    def _calculate_idf(self):
        """计算逆文档频率(IDF)"""
        try:
            for term, doc_list in self.inverted_index.items():
                # IDF = log(总文档数 / 包含该词项的文档数)
                doc_count = len(doc_list)
                if doc_count > 0:
                    self.idf[term] = math.log(self.total_docs / doc_count) + 1  # +1避免IDF为0
            
            # 为所有在文档中出现但不在inverted_index中的词项设置默认IDF
            for doc_id, terms in self.document_terms.items():
                for term in terms:
                    if term not in self.idf:
                        self.idf[term] = 1.0  # 默认值
            
        except Exception as e:
            logger.error(f"计算IDF失败: {str(e)}")
    
    def search(self, query: str, top_k: int = 10, include_score: bool = False) -> List[Dict[str, Any]]:
        """搜索文档
        
        Args:
            query: 搜索查询
            top_k: 返回的结果数量
            include_score: 是否包含得分
        
        Returns:
            搜索结果列表
        """
        try:
            # 如果索引为空，直接返回空结果
            if self.total_docs == 0:
                return []
            
            # 对查询分词
            query_tokens = self.tokenize(query)
            if not query_tokens:
                return []
            
            # 计算查询向量
            query_vector = Counter(query_tokens)
            query_norm = math.sqrt(sum(count ** 2 for count in query_vector.values()))
            
            # 计算文档得分
            doc_scores = defaultdict(float)
            
            # 遍历查询中的每个词项
            for term, query_tf in query_vector.items():
                # 如果词项不在索引中，跳过
                if term not in self.inverted_index:
                    continue
                
                # 获取包含该词项的文档
                for doc_id, _ in self.inverted_index[term]:
                    # 计算文档的TF-IDF
                    doc_tf = self.document_terms[doc_id].get(term, 0) / max(self.document_terms[doc_id].values(), default=1)
                    idf = self.idf.get(term, 1.0)
                    tf_idf = doc_tf * idf
                    
                    # 计算余弦相似度的分子部分
                    doc_scores[doc_id] += query_tf * tf_idf
            
            # 对得分进行归一化
            for doc_id in doc_scores:
                # 计算文档向量的模长
                doc_norm = math.sqrt(sum((tf * self.idf.get(term, 1.0)) ** 2 
                                         for term, tf in self.document_terms[doc_id].items()))
                
                if doc_norm > 0 and query_norm > 0:
                    doc_scores[doc_id] = doc_scores[doc_id] / (query_norm * doc_norm)
            
            # 按得分排序并获取前top_k个结果
            sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            # 构建结果列表
            results = []
            for doc_id, score in sorted_results:
                result = {'id': doc_id}
                
                # 如果有元数据管理器，获取元数据
                if self.metadata_manager:
                    metadata = self.metadata_manager.get_knowledge_item(doc_id)
                    if metadata:
                        result.update(metadata)
                
                # 是否包含得分
                if include_score:
                    result['score'] = score
                
                results.append(result)
            
            logger.info(f"搜索完成: '{query}'，找到 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return []
    
    def advanced_search(self, 
                        query: str = None, 
                        title: str = None, 
                        content: str = None, 
                        tags: List[str] = None, 
                        category: str = None, 
                        top_k: int = 10) -> List[Dict[str, Any]]:
        """高级搜索功能
        
        Args:
            query: 全文搜索查询
            title: 标题搜索
            content: 内容搜索
            tags: 标签筛选
            category: 分类筛选
            top_k: 返回的结果数量
        
        Returns:
            搜索结果列表
        """
        try:
            # 构建复合查询
            combined_query = []
            if query:
                combined_query.append(query)
            if title:
                combined_query.append(title)
            if content:
                combined_query.append(content)
            
            # 执行全文搜索
            full_text_query = ' '.join(combined_query)
            results = self.search(full_text_query, top_k=100, include_score=True)  # 获取更多结果以便筛选
            
            # 如果有元数据管理器，进行标签和分类筛选
            if self.metadata_manager and (tags or category):
                filtered_results = []
                
                for result in results:
                    match = True
                    
                    # 标签筛选
                    if tags:
                        result_tags = set(tag.lower() for tag in result.get('tags', []))
                        search_tags = set(tag.lower() for tag in tags)
                        if not search_tags.intersection(result_tags):
                            match = False
                    
                    # 分类筛选
                    if category and result.get('category', '').lower() != category.lower():
                        match = False
                    
                    if match:
                        filtered_results.append(result)
                
                results = filtered_results
            
            # 按得分排序并限制数量
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            results = results[:top_k]
            
            logger.info(f"高级搜索完成，找到 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"高级搜索失败: {str(e)}")
            return []
    
    def load_index(self):
        """从文件加载索引"""
        try:
            if os.path.exists(self.index_file):
                with open(self.index_file, 'rb') as f:
                    data = pickle.load(f)
                    self.inverted_index = defaultdict(list, data.get('inverted_index', {}))
                    self.document_terms = defaultdict(Counter, data.get('document_terms', {}))
                    self.idf = data.get('idf', {})
                    self.total_docs = data.get('total_docs', 0)
                logger.info(f"成功加载搜索索引，包含 {self.total_docs} 个文档")
            else:
                logger.info("搜索索引文件不存在，将创建新的索引")
        except Exception as e:
            logger.error(f"加载搜索索引失败: {str(e)}")
            # 重置索引
            self.inverted_index = defaultdict(list)
            self.document_terms = defaultdict(Counter)
            self.idf = {}
            self.total_docs = 0
    
    def save_index(self):
        """保存索引到文件"""
        try:
            data = {
                'inverted_index': dict(self.inverted_index),  # 转换为普通字典以支持pickle
                'document_terms': {doc_id: dict(terms) for doc_id, terms in self.document_terms.items()},
                'idf': self.idf,
                'total_docs': self.total_docs
            }
            
            with open(self.index_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"成功保存搜索索引到 {self.index_file}")
        except Exception as e:
            logger.error(f"保存搜索索引失败: {str(e)}")
    
    def rebuild_index(self, documents: List[Dict[str, Any]]):
        """重建索引
        
        Args:
            documents: 文档列表，每个文档包含id, title, content字段
        """
        try:
            # 重置索引
            self.inverted_index = defaultdict(list)
            self.document_terms = defaultdict(Counter)
            self.idf = {}
            self.total_docs = 0
            
            # 添加所有文档
            for doc in documents:
                self.add_document(doc['id'], doc.get('title', ''), doc.get('content', ''))
            
            # 保存索引
            self.save_index()
            
            logger.info(f"成功重建索引，包含 {self.total_docs} 个文档")
        except Exception as e:
            logger.error(f"重建索引失败: {str(e)}")
    
    def get_index_size(self) -> Dict[str, Any]:
        """获取索引大小信息
        
        Returns:
            索引大小信息字典
        """
        try:
            size_info = {
                'document_count': self.total_docs,
                'unique_terms': len(self.inverted_index),
                'average_terms_per_doc': sum(len(terms) for terms in self.document_terms.values()) / self.total_docs if self.total_docs > 0 else 0
            }
            
            # 如果索引文件存在，获取文件大小
            if os.path.exists(self.index_file):
                size_info['index_file_size_bytes'] = os.path.getsize(self.index_file)
                size_info['index_file_size_mb'] = size_info['index_file_size_bytes'] / (1024 * 1024)
            
            return size_info
        except Exception as e:
            logger.error(f"获取索引大小信息失败: {str(e)}")
            return {
                'document_count': 0,
                'unique_terms': 0,
                'average_terms_per_doc': 0
            }
    
    def optimize_index(self):
        """优化索引"""
        try:
            # 移除低频率词项（在少于3个文档中出现的词项）
            low_freq_terms = []
            for term, doc_list in self.inverted_index.items():
                if len(doc_list) < 3:
                    low_freq_terms.append(term)
            
            for term in low_freq_terms:
                del self.inverted_index[term]
                if term in self.idf:
                    del self.idf[term]
            
            # 重新计算IDF
            self._calculate_idf()
            
            # 保存优化后的索引
            self.save_index()
            
            logger.info(f"成功优化索引，移除了 {len(low_freq_terms)} 个低频率词项")
        except Exception as e:
            logger.error(f"优化索引失败: {str(e)}")
    
    def backup_index(self, backup_path: Optional[str] = None) -> Optional[str]:
        """备份索引
        
        Args:
            backup_path: 备份路径，如果不提供则使用默认路径
        
        Returns:
            备份文件路径或None
        """
        try:
            if not os.path.exists(self.index_file):
                logger.warning("索引文件不存在，无法备份")
                return None
            
            if not backup_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = os.path.join(self.storage_path, f'search_index_backup_{timestamp}.pkl')
            
            # 确保备份目录存在
            backup_dir = os.path.dirname(backup_path)
            os.makedirs(backup_dir, exist_ok=True)
            
            # 复制索引文件
            with open(self.index_file, 'rb') as f_src:
                with open(backup_path, 'wb') as f_dst:
                    f_dst.write(f_src.read())
            
            logger.info(f"成功备份索引到 {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"备份索引失败: {str(e)}")
            return None
    
    def restore_index(self, backup_path: str) -> bool:
        """从备份恢复索引
        
        Args:
            backup_path: 备份文件路径
        
        Returns:
            是否成功恢复
        """
        try:
            if not os.path.exists(backup_path):
                logger.warning(f"备份文件不存在: {backup_path}")
                return False
            
            # 先备份当前索引
            current_backup = self.backup_index(backup_path + '.current')
            
            # 复制备份文件
            with open(backup_path, 'rb') as f_src:
                with open(self.index_file, 'wb') as f_dst:
                    f_dst.write(f_src.read())
            
            # 重新加载索引
            self.load_index()
            
            logger.info(f"成功从备份恢复索引: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"恢复索引失败: {str(e)}")
            # 尝试恢复原始状态
            if os.path.exists(backup_path + '.current'):
                try:
                    with open(backup_path + '.current', 'rb') as f_src:
                        with open(self.index_file, 'wb') as f_dst:
                            f_dst.write(f_src.read())
                    self.load_index()
                except:
                    pass
            return False

# 示例用法
if __name__ == '__main__':
    # 创建临时测试路径
    test_path = os.path.join(os.environ.get('TEMP', '/tmp'), 'self_brain_search_test')
    
    # 初始化搜索引擎
    search_engine = KnowledgeSearchEngine(test_path)
    
    # 添加测试文档
    test_docs = [
        {
            'id': 'doc_001',
            'title': 'Python 编程入门',
            'content': 'Python 是一种简单易学的编程语言，广泛应用于数据分析、人工智能等领域。'
        },
        {
            'id': 'doc_002',
            'title': '机器学习基础',
            'content': '机器学习是人工智能的一个分支，通过算法使计算机能够从数据中学习并做出预测。'
        },
        {
            'id': 'doc_003',
            'title': '深度学习入门',
            'content': '深度学习是机器学习的一个子集，使用神经网络模型来处理复杂的数据模式。'
        }
    ]
    
    # 添加文档到索引
    for doc in test_docs:
        search_engine.add_document(doc['id'], doc['title'], doc['content'])
    
    # 保存索引
    search_engine.save_index()
    
    # 搜索测试
    results = search_engine.search('Python 编程')
    print(f"搜索结果: {json.dumps(results, ensure_ascii=False, indent=2)}")
    
    # 高级搜索测试
    advanced_results = search_engine.advanced_search(title='入门')
    print(f"高级搜索结果: {json.dumps(advanced_results, ensure_ascii=False, indent=2)}")
    
    # 获取索引信息
    index_info = search_engine.get_index_size()
    print(f"索引信息: {json.dumps(index_info, ensure_ascii=False, indent=2)}")
    
    # 清理测试数据
    if os.path.exists(test_path):
        import shutil
        shutil.rmtree(test_path)
        print(f"已清理测试数据: {test_path}")