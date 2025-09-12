# -*- coding: utf-8 -*-
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0

"""
统一知识库模型 | Unified Knowledge Base Model
整合标准模式和增强模式功能
"""

import json
import os
import sqlite3
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
from datetime import datetime
import hashlib

class KnowledgeEntry:
    """知识条目数据结构"""
    
    def __init__(self, content: str, category: str = "general", 
                 metadata: Dict[str, Any] = None, tags: List[str] = None):
        self.content = content
        self.category = category
        self.metadata = metadata or {}
        self.tags = tags or []
        self.id = self._generate_id()
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.access_count = 0
        self.relevance_score = 1.0
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.category}_{content_hash}_{int(time.time())}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "relevance_score": self.relevance_score
        }

class UnifiedKnowledgeModel:
    """
    统一知识库模型
    支持知识存储、检索、更新和分析
    """
    
    def __init__(self, mode: str = "standard", config: Optional[Dict] = None):
        self.mode = mode
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 数据库配置
        self.db_path = self.config.get("db_path", "knowledge_base.db")
        self.memory_mode = self.config.get("memory_mode", False)
        
        # 知识存储
        self.knowledge_base = {}
        self.categories = set()
        self.search_index = {}
        
        # 性能优化
        self.cache = {}
        self.cache_size = 1000
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 初始化
        self._initialize_database()
        self._load_existing_knowledge()
        
    def _initialize_database(self):
        """初始化数据库"""
        if self.memory_mode:
            self.conn = sqlite3.connect(":memory:")
        else:
            self.conn = sqlite3.connect(self.db_path)
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT,
                metadata TEXT,
                tags TEXT,
                created_at TEXT,
                updated_at TEXT,
                access_count INTEGER DEFAULT 0,
                relevance_score REAL DEFAULT 1.0
            )
        ''')
        
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_category ON knowledge(category);
        ''')
        
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_tags ON knowledge(tags);
        ''')
        
        self.conn.commit()
    
    def _load_existing_knowledge(self):
        """加载现有知识"""
        try:
            cursor = self.conn.execute('SELECT * FROM knowledge')
            for row in cursor.fetchall():
                entry = KnowledgeEntry(
                    content=row[1],
                    category=row[2],
                    metadata=json.loads(row[3]) if row[3] else {},
                    tags=json.loads(row[4]) if row[4] else []
                )
                entry.id = row[0]
                entry.created_at = row[5]
                entry.updated_at = row[6]
                entry.access_count = row[7]
                entry.relevance_score = row[8]
                
                self.knowledge_base[entry.id] = entry
                self.categories.add(entry.category)
        except Exception as e:
            self.logger.error(f"加载知识库失败: {e}")
    
    def add_knowledge(self, content: str, category: str = "general", 
                     metadata: Dict[str, Any] = None, tags: List[str] = None) -> str:
        """添加新知识"""
        with self._lock:
            entry = KnowledgeEntry(content, category, metadata, tags)
            
            # 增强模式下的额外处理
            if self.mode == "enhanced":
                entry = self._enhanced_processing(entry)
            
            self.knowledge_base[entry.id] = entry
            self.categories.add(category)
            
            # 保存到数据库
            self._save_to_database(entry)
            
            # 更新索引
            self._update_search_index(entry)
            
            return entry.id
    
    def search_knowledge(self, query: str, category: str = None, 
                        limit: int = 10) -> List[Dict[str, Any]]:
        """搜索知识"""
        with self._lock:
            results = []
            
            # 简单文本匹配
            query_lower = query.lower()
            for entry in self.knowledge_base.values():
                if category and entry.category != category:
                    continue
                
                score = 0
                if query_lower in entry.content.lower():
                    score += 2
                if any(query_lower in tag.lower() for tag in entry.tags):
                    score += 1
                if query_lower in entry.category.lower():
                    score += 1
                
                if score > 0:
                    entry.access_count += 1
                    results.append({
                        "entry": entry.to_dict(),
                        "score": score * entry.relevance_score
                    })
            
            # 增强模式下的语义搜索
            if self.mode == "enhanced":
                results = self._semantic_search(query, results)
            
            # 排序并返回
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]
    
    def get_knowledge_by_category(self, category: str) -> List[Dict[str, Any]]:
        """按类别获取知识"""
        with self._lock:
            return [
                entry.to_dict() 
                for entry in self.knowledge_base.values() 
                if entry.category == category
            ]
    
    def update_knowledge(self, entry_id: str, **kwargs) -> bool:
        """更新知识条目"""
        with self._lock:
            if entry_id not in self.knowledge_base:
                return False
            
            entry = self.knowledge_base[entry_id]
            
            if "content" in kwargs:
                entry.content = kwargs["content"]
            if "category" in kwargs:
                entry.category = kwargs["category"]
            if "metadata" in kwargs:
                entry.metadata.update(kwargs["metadata"])
            if "tags" in kwargs:
                entry.tags = kwargs["tags"]
            
            entry.updated_at = datetime.now().isoformat()
            
            # 更新数据库
            self._save_to_database(entry)
            
            return True
    
    def delete_knowledge(self, entry_id: str) -> bool:
        """删除知识条目"""
        with self._lock:
            if entry_id not in self.knowledge_base:
                return False
            
            entry = self.knowledge_base.pop(entry_id)
            
            # 从数据库删除
            self.conn.execute('DELETE FROM knowledge WHERE id = ?', (entry_id,))
            self.conn.commit()
            
            return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计"""
        with self._lock:
            total_entries = len(self.knowledge_base)
            if total_entries == 0:
                return {"total_entries": 0}
            
            category_counts = {}
            for entry in self.knowledge_base.values():
                category_counts[entry.category] = category_counts.get(entry.category, 0) + 1
            
            avg_access_count = np.mean([e.access_count for e in self.knowledge_base.values()])
            
            return {
                "total_entries": total_entries,
                "categories": list(self.categories),
                "category_counts": category_counts,
                "average_access_count": avg_access_count,
                "mode": self.mode
            }
    
    def _enhanced_processing(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        """增强模式下的额外处理"""
        # 自动生成标签
        if not entry.tags:
            words = entry.content.lower().split()
            entry.tags = list(set([w for w in words if len(w) > 3][:5]))
        
        # 计算相关性分数
        if len(entry.content) > 50:
            entry.relevance_score = min(2.0, 1.0 + len(entry.content) / 200)
        
        # 添加语义分析
        entry.metadata["word_count"] = len(entry.content.split())
        entry.metadata["char_count"] = len(entry.content)
        
        return entry
    
    def _semantic_search(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """语义搜索（增强模式）"""
        # 简单的关键词扩展
        query_words = query.lower().split()
        for result in results:
            entry = result["entry"]
            content_words = entry["content"].lower().split()
            
            # 计算词频相似度
            common_words = set(query_words) & set(content_words)
            if common_words:
                result["score"] += len(common_words) * 0.5
        
        return results
    
    def _save_to_database(self, entry: KnowledgeEntry):
        """保存到数据库"""
        self.conn.execute('''
            INSERT OR REPLACE INTO knowledge 
            (id, content, category, metadata, tags, created_at, updated_at, access_count, relevance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.id, entry.content, entry.category,
            json.dumps(entry.metadata), json.dumps(entry.tags),
            entry.created_at, entry.updated_at, entry.access_count, entry.relevance_score
        ))
        self.conn.commit()
    
    def _update_search_index(self, entry: KnowledgeEntry):
        """更新搜索索引"""
        # 简单的关键词索引
        words = entry.content.lower().split()
        for word in words:
            if word not in self.search_index:
                self.search_index[word] = []
            if entry.id not in self.search_index[word]:
                self.search_index[word].append(entry.id)
    
    def export_knowledge(self, format: str = "json") -> str:
        """导出知识库"""
        with self._lock:
            if format == "json":
                return json.dumps([entry.to_dict() for entry in self.knowledge_base.values()], 
                                ensure_ascii=False, indent=2)
            return ""
    
    def import_knowledge(self, data: str, format: str = "json") -> int:
        """导入知识库"""
        try:
            if format == "json":
                entries = json.loads(data)
                count = 0
                for entry_data in entries:
                    self.add_knowledge(
                        content=entry_data["content"],
                        category=entry_data.get("category", "general"),
                        metadata=entry_data.get("metadata", {}),
                        tags=entry_data.get("tags", [])
                    )
                    count += 1
                return count
        except Exception as e:
            self.logger.error(f"导入知识库失败: {e}")
            return 0
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()