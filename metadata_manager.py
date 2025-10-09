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
import time
import shutil
import logging
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MetadataManager")

class KnowledgeMetadataManager:
    """管理知识库的元数据信息"""
    
    def __init__(self, storage_path):
        """初始化知识库元数据管理器
        
        Args:
            storage_path: 知识库存储路径
        """
        self.storage_path = storage_path
        self.metadata_file = os.path.join(storage_path, 'metadata.json')
        self.metadata = {}
        
        # 确保存储路径存在
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            logger.info(f"知识库存储路径已创建或已存在: {self.storage_path}")
        except Exception as e:
            logger.error(f"创建知识库存储路径失败: {str(e)}")
            # 创建临时目录作为回退
            self.storage_path = os.path.join(os.environ.get('TEMP', '/tmp'), 'self_brain_knowledge_temp')
            os.makedirs(self.storage_path, exist_ok=True)
            self.metadata_file = os.path.join(self.storage_path, 'metadata.json')
            logger.warning(f"使用临时存储路径作为回退: {self.storage_path}")
        
        # 加载现有元数据
        self.load_metadata()
    
    def load_metadata(self):
        """从文件加载元数据"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"成功加载元数据，共 {len(self.metadata)} 条记录")
            else:
                logger.info("元数据文件不存在，将创建新的元数据")
        except json.JSONDecodeError:
            logger.error("元数据文件格式错误，创建新的元数据")
            self.metadata = {}
        except Exception as e:
            logger.error(f"加载元数据失败: {str(e)}")
            self.metadata = {}
    
    def save_metadata(self):
        """保存元数据到文件"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"成功保存元数据到 {self.metadata_file}")
        except Exception as e:
            logger.error(f"保存元数据失败: {str(e)}")
    
    def add_knowledge_item(self, knowledge_id, title, content, source=None, tags=None, category=None):
        """添加知识项到元数据
        
        Args:
            knowledge_id: 知识项ID
            title: 知识项标题
            content: 知识项内容
            source: 知识项来源
            tags: 标签列表
            category: 分类
        """
        try:
            # 计算内容哈希值用于完整性检查
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            # 构建元数据项
            metadata_item = {
                'id': knowledge_id,
                'title': title,
                'content_preview': content[:200] + '...' if len(content) > 200 else content,
                'source': source or 'unknown',
                'tags': tags or [],
                'category': category or 'general',
                'content_hash': content_hash,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'status': 'active',
                'access_count': 0
            }
            
            # 更新元数据
            self.metadata[knowledge_id] = metadata_item
            self.save_metadata()
            logger.info(f"成功添加知识项: {title}")
            return True
        except Exception as e:
            logger.error(f"添加知识项失败: {str(e)}")
            return False
    
    def update_knowledge_item(self, knowledge_id, **kwargs):
        """更新知识项元数据
        
        Args:
            knowledge_id: 知识项ID
            **kwargs: 要更新的字段
        """
        try:
            if knowledge_id not in self.metadata:
                logger.warning(f"知识项不存在: {knowledge_id}")
                return False
            
            # 更新字段
            self.metadata[knowledge_id].update(kwargs)
            self.metadata[knowledge_id]['updated_at'] = datetime.now().isoformat()
            self.save_metadata()
            logger.info(f"成功更新知识项元数据: {knowledge_id}")
            return True
        except Exception as e:
            logger.error(f"更新知识项元数据失败: {str(e)}")
            return False
    
    def delete_knowledge_item(self, knowledge_id):
        """删除知识项元数据
        
        Args:
            knowledge_id: 知识项ID
        """
        try:
            if knowledge_id not in self.metadata:
                logger.warning(f"知识项不存在: {knowledge_id}")
                return False
            
            del self.metadata[knowledge_id]
            self.save_metadata()
            logger.info(f"成功删除知识项元数据: {knowledge_id}")
            return True
        except Exception as e:
            logger.error(f"删除知识项元数据失败: {str(e)}")
            return False
    
    def get_knowledge_item(self, knowledge_id):
        """获取知识项元数据
        
        Args:
            knowledge_id: 知识项ID
        
        Returns:
            元数据字典或None
        """
        try:
            if knowledge_id in self.metadata:
                # 增加访问计数
                self.metadata[knowledge_id]['access_count'] += 1
                # 延迟保存，避免频繁写入
                if self.metadata[knowledge_id]['access_count'] % 10 == 0:
                    self.save_metadata()
                return self.metadata[knowledge_id]
            return None
        except Exception as e:
            logger.error(f"获取知识项元数据失败: {str(e)}")
            return None
    
    def search_knowledge_items(self, keyword=None, tags=None, category=None, status=None, limit=100):
        """搜索知识项
        
        Args:
            keyword: 关键词
            tags: 标签列表
            category: 分类
            status: 状态
            limit: 结果数量限制
        
        Returns:
            匹配的元数据列表
        """
        try:
            results = []
            
            for item in self.metadata.values():
                # 检查搜索条件
                match = True
                
                if keyword and keyword.lower() not in item['title'].lower() and keyword.lower() not in item['content_preview'].lower():
                    match = False
                
                if tags:
                    item_tags = set(tag.lower() for tag in item['tags'])
                    search_tags = set(tag.lower() for tag in tags)
                    if not search_tags.issubset(item_tags):
                        match = False
                
                if category and item['category'].lower() != category.lower():
                    match = False
                
                if status and item['status'].lower() != status.lower():
                    match = False
                
                if match:
                    results.append(item)
            
            # 按更新时间排序
            results.sort(key=lambda x: x['updated_at'], reverse=True)
            
            return results[:limit]
        except Exception as e:
            logger.error(f"搜索知识项失败: {str(e)}")
            return []
    
    def get_all_knowledge_items(self, limit=None):
        """获取所有知识项
        
        Args:
            limit: 结果数量限制
        
        Returns:
            元数据列表
        """
        try:
            items = list(self.metadata.values())
            # 按更新时间排序
            items.sort(key=lambda x: x['updated_at'], reverse=True)
            
            if limit:
                return items[:limit]
            return items
        except Exception as e:
            logger.error(f"获取所有知识项失败: {str(e)}")
            return []
    
    def get_statistics(self):
        """获取知识库统计信息
        
        Returns:
            统计信息字典
        """
        try:
            stats = {
                'total_items': len(self.metadata),
                'categories': {},
                'tags': {},
                'active_items': 0,
                'total_access_count': 0
            }
            
            for item in self.metadata.values():
                # 统计分类
                category = item['category']
                if category not in stats['categories']:
                    stats['categories'][category] = 0
                stats['categories'][category] += 1
                
                # 统计标签
                for tag in item['tags']:
                    if tag not in stats['tags']:
                        stats['tags'][tag] = 0
                    stats['tags'][tag] += 1
                
                # 统计活跃项
                if item['status'] == 'active':
                    stats['active_items'] += 1
                
                # 统计总访问次数
                stats['total_access_count'] += item['access_count']
            
            # 按数量排序分类和标签
            stats['categories'] = dict(sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True))
            stats['tags'] = dict(sorted(stats['tags'].items(), key=lambda x: x[1], reverse=True))
            
            return stats
        except Exception as e:
            logger.error(f"获取知识库统计信息失败: {str(e)}")
            return {
                'total_items': 0,
                'categories': {},
                'tags': {},
                'active_items': 0,
                'total_access_count': 0
            }
    
    def backup_metadata(self, backup_path=None):
        """备份元数据
        
        Args:
            backup_path: 备份路径，如果不提供则使用默认路径
        
        Returns:
            备份文件路径或None
        """
        try:
            if not backup_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = os.path.join(self.storage_path, f'metadata_backup_{timestamp}.json')
            
            # 确保备份目录存在
            backup_dir = os.path.dirname(backup_path)
            os.makedirs(backup_dir, exist_ok=True)
            
            # 复制元数据文件
            shutil.copy2(self.metadata_file, backup_path)
            logger.info(f"成功备份元数据到 {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"备份元数据失败: {str(e)}")
            return None
    
    def restore_metadata(self, backup_path):
        """从备份恢复元数据
        
        Args:
            backup_path: 备份文件路径
        
        Returns:
            是否成功恢复
        """
        try:
            if not os.path.exists(backup_path):
                logger.warning(f"备份文件不存在: {backup_path}")
                return False
            
            # 先备份当前元数据
            self.backup_metadata(backup_path + '.current')
            
            # 复制备份文件
            shutil.copy2(backup_path, self.metadata_file)
            
            # 重新加载元数据
            self.load_metadata()
            
            logger.info(f"成功从备份恢复元数据: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"恢复元数据失败: {str(e)}")
            # 尝试恢复原始状态
            try:
                shutil.copy2(backup_path + '.current', self.metadata_file)
                self.load_metadata()
            except:
                pass
            return False
    
    def clean_inactive_items(self, days=90):
        """清理非活跃的知识项
        
        Args:
            days: 多少天未更新的项目被视为非活跃
        
        Returns:
            清理的项目数量
        """
        try:
            cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
            cleaned_count = 0
            
            for knowledge_id, item in list(self.metadata.items()):
                updated_time = datetime.fromisoformat(item['updated_at']).timestamp()
                if updated_time < cutoff_time and item['access_count'] < 5:
                    # 标记为非活跃，而不是直接删除
                    self.update_knowledge_item(knowledge_id, status='inactive')
                    cleaned_count += 1
            
            logger.info(f"成功清理 {cleaned_count} 个非活跃知识项")
            return cleaned_count
        except Exception as e:
            logger.error(f"清理非活跃知识项失败: {str(e)}")
            return 0

# 示例用法
if __name__ == '__main__':
    # 创建临时测试路径
    test_path = os.path.join(os.environ.get('TEMP', '/tmp'), 'self_brain_test')
    
    # 初始化管理器
    manager = KnowledgeMetadataManager(test_path)
    
    # 添加测试知识项
    manager.add_knowledge_item(
        'test_001',
        '测试知识项',
        '这是一个测试知识项的内容。\n包含多行文本。\n用于测试MetadataManager的功能。',
        source='测试源',
        tags=['测试', '示例'],
        category='测试分类'
    )
    
    # 获取统计信息
    stats = manager.get_statistics()
    print(f"统计信息: {json.dumps(stats, ensure_ascii=False, indent=2)}")
    
    # 搜索知识项
    results = manager.search_knowledge_items(keyword='测试')
    print(f"搜索结果: {json.dumps(results, ensure_ascii=False, indent=2)}")
    
    # 备份元数据
    backup_path = manager.backup_metadata()
    print(f"备份路径: {backup_path}")
    
    # 清理测试数据
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
        print(f"已清理测试数据: {test_path}")