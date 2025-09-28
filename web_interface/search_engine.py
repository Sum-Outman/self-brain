import os
import json
import time
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, KEYWORD, ID, DATETIME, NUMERIC
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import Every, Term, And, Or
from whoosh.analysis import StemmingAnalyzer
import jieba
import re
from pathlib import Path
import shutil
import tempfile
import logging
from datetime import datetime

class KnowledgeSearchEngine:
    def __init__(self, index_dir="search_index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # 定义索引模式
        self.schema = Schema(
            id=ID(stored=True, unique=True),
            title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            description=TEXT(stored=True),
            tags=KEYWORD(stored=True, lowercase=True, commas=True),
            category=KEYWORD(stored=True, lowercase=True),
            file_type=KEYWORD(stored=True, lowercase=True),
            file_path=ID(stored=True),
            file_size=NUMERIC(stored=True),
            created_at=DATETIME(stored=True),
            updated_at=DATETIME(stored=True),
            author=KEYWORD(stored=True),
            language=KEYWORD(stored=True),
            checksum=ID(stored=True)
        )
        
        self._create_or_open_index()
    
    def _create_or_open_index(self):
        """创建或打开索引，处理pickle协议版本和文件锁定问题"""
        max_retries = 3
        retry_delay = 1  # 秒
        
        # 配置日志记录
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(__name__)
        
        for attempt in range(max_retries):
            try:
                # 确保目录存在
                if not self.index_dir.exists():
                    self.index_dir.mkdir(parents=True, exist_ok=True)
                
                # 尝试打开现有索引或创建新索引
                if exists_in(str(self.index_dir)):
                    try:
                        self.ix = open_dir(str(self.index_dir))
                        self.logger.info(f"成功打开现有索引: {self.index_dir}")
                        return  # 成功打开索引，退出方法
                    except ValueError as e:
                        if "unsupported pickle protocol" in str(e).lower():
                            self.logger.warning(f"检测到不支持的pickle协议版本。重新创建索引: {self.index_dir}...")
                            # 尝试备份并重新创建索引
                            temp_backup = None
                            try:
                                # 创建临时备份目录
                                temp_backup = tempfile.mkdtemp(prefix="search_index_backup_")
                                backup_path = Path(temp_backup)
                                # 移动文件而不是直接删除以提高安全性
                                for filename in os.listdir(str(self.index_dir)):
                                    src = self.index_dir / filename
                                    dst = backup_path / filename
                                    shutil.move(str(src), str(dst))
                                self.logger.info(f"已备份旧索引文件到: {temp_backup}")
                            except Exception as backup_error:
                                self.logger.error(f"备份旧索引失败: {backup_error}")
                                # 即使备份失败，也要继续创建新索引
                            # 创建新索引
                            self.ix = create_in(str(self.index_dir), self.schema)
                            self.logger.info(f"创建了兼容pickle协议的新索引: {self.index_dir}")
                            return  # 成功创建索引，退出方法
                        else:
                            raise
                    except PermissionError as e:
                        self.logger.warning(f"访问索引文件时权限错误 (尝试 {attempt+1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        # 最后一次尝试权限错误 - 使用临时目录
                        self.logger.error("所有尝试都失败了。创建临时索引目录...")
                        temp_index_dir = Path(tempfile.gettempdir()) / f"self_brain_temp_index_{int(time.time())}"
                        self.logger.info(f"使用临时索引目录: {temp_index_dir}")
                        self.index_dir = temp_index_dir  # 更新index_dir属性
                        if not temp_index_dir.exists():
                            temp_index_dir.mkdir(parents=True, exist_ok=True)
                        self.ix = create_in(str(temp_index_dir), self.schema)
                        self.logger.info(f"在临时目录创建索引: {temp_index_dir}")
                        return  # 成功创建临时索引
                else:
                    # 如果索引不存在，创建新索引
                    self.ix = create_in(str(self.index_dir), self.schema)
                    self.logger.info(f"创建新索引: {self.index_dir}")
                    return  # 成功创建索引，退出方法
                
            except Exception as e:
                self.logger.error(f"创建或打开索引时出错 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                # 所有尝试都失败 - 作为最后的手段使用临时目录
                self.logger.error("所有尝试都失败了。创建临时索引目录...")
                temp_index_dir = Path(tempfile.gettempdir()) / f"self_brain_temp_index_{int(time.time())}"
                self.logger.info(f"使用临时索引目录: {temp_index_dir}")
                self.index_dir = temp_index_dir  # 更新index_dir属性
                if not temp_index_dir.exists():
                    temp_index_dir.mkdir(parents=True, exist_ok=True)
                self.ix = create_in(str(temp_index_dir), self.schema)
                self.logger.info(f"在临时目录创建索引: {temp_index_dir}")
                return  # 成功创建临时索引
        
        # 如果所有重试都失败，抛出异常
        raise Exception(f"经过{max_retries}次尝试后，无法创建或打开索引")
    
    def _safe_delete_index_files(self, index_dir):
        """安全删除索引文件，处理各种可能的错误"""
        # 配置日志记录
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(__name__)
            
        try:
            if not isinstance(index_dir, Path):
                index_dir = Path(index_dir)
                
            for filename in os.listdir(str(index_dir)):
                file_path = index_dir / filename
                try:
                    if file_path.is_file():
                        # 对于Windows，文件有时会被锁定 - 尝试多次删除
                        max_delete_retries = 3
                        for delete_attempt in range(max_delete_retries):
                            try:
                                file_path.unlink()
                                self.logger.debug(f"已删除索引文件: {file_path}")
                                break
                            except PermissionError as e:
                                self.logger.warning(f"删除{file_path}时权限被拒绝 (尝试 {delete_attempt+1}/{max_delete_retries}): {e}")
                                if delete_attempt < max_delete_retries - 1:
                                    time.sleep(0.5)
                                    continue
                                # 如果最后一次尝试失败，移至备份
                                backup_dir = Path(tempfile.gettempdir()) / f"index_backup_{int(time.time())}"
                                backup_dir.mkdir(parents=True, exist_ok=True)
                                backup_path = backup_dir / filename
                                try:
                                    shutil.move(str(file_path), str(backup_path))
                                    self.logger.info(f"已将锁定的文件移至备份: {backup_path}")
                                except Exception as move_error:
                                    self.logger.error(f"无法移动锁定的文件: {move_error}")
                except Exception as e:
                    self.logger.error(f"删除{file_path}时出错: {e}")
        except Exception as e:
            self.logger.error(f"访问索引目录{index_dir}时出错: {e}")
    
    def _preprocess_chinese_text(self, text):
        """预处理中文文本"""
        if not text:
            return ""
        
        # 使用jieba分词
        words = jieba.cut(text)
        return " ".join(words)
    
    def _extract_text_content(self, file_path):
        """提取文件文本内容"""
        file_path = Path(file_path)
        
        try:
            if file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif file_path.suffix.lower() == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 移除Markdown标记
                    content = re.sub(r'#+\s*', '', content)
                    content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)
                    content = re.sub(r'\*(.*?)\*', r'\1', content)
                    return content
            
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2)
            
            elif file_path.suffix.lower() in ['.py', '.js']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 移除注释
                    content = re.sub(r'#.*', '', content)
                    content = re.sub(r'//.*', '', content)
                    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
                    return content
            
            else:
                return f"Binary file: {file_path.suffix}"
                
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def add_document(self, metadata):
        """添加文档到索引"""
        writer = self.ix.writer()
        
        file_path = Path("d:\\shiyan\\knowledge_base_storage") / metadata.get('file_path', '')
        content = self._extract_text_content(file_path)
        
        # 预处理内容
        processed_content = self._preprocess_chinese_text(content)
        processed_title = self._preprocess_chinese_text(metadata.get('title', ''))
        processed_description = self._preprocess_chinese_text(metadata.get('description', ''))
        
        writer.add_document(
            id=metadata.get('id'),
            title=processed_title,
            content=processed_content,
            description=processed_description,
            tags=",".join(metadata.get('tags', [])),
            category=metadata.get('category', ''),
            file_type=metadata.get('file_extension', ''),
            file_path=metadata.get('file_path', ''),
            file_size=metadata.get('file_size', 0),
            created_at=metadata.get('created_at'),
            updated_at=metadata.get('updated_at'),
            author=metadata.get('author', ''),
            language=metadata.get('language', ''),
            checksum=metadata.get('checksum', '')
        )
        
        writer.commit()
    
    def update_document(self, metadata):
        """更新索引中的文档"""
        writer = self.ix.writer()
        
        # 先删除旧文档
        writer.delete_by_term('id', metadata.get('id'))
        
        # 添加更新后的文档
        file_path = Path("d:\\shiyan\\knowledge_base_storage") / metadata.get('file_path', '')
        content = self._extract_text_content(file_path)
        
        processed_content = self._preprocess_chinese_text(content)
        processed_title = self._preprocess_chinese_text(metadata.get('title', ''))
        processed_description = self._preprocess_chinese_text(metadata.get('description', ''))
        
        writer.add_document(
            id=metadata.get('id'),
            title=processed_title,
            content=processed_content,
            description=processed_description,
            tags=",".join(metadata.get('tags', [])),
            category=metadata.get('category', ''),
            file_type=metadata.get('file_extension', ''),
            file_path=metadata.get('file_path', ''),
            file_size=metadata.get('file_size', 0),
            created_at=metadata.get('created_at'),
            updated_at=metadata.get('updated_at'),
            author=metadata.get('author', ''),
            language=metadata.get('language', ''),
            checksum=metadata.get('checksum', '')
        )
        
        writer.commit()
    
    def search(self, query_string, category=None, tags=None, limit=50):
        """搜索文档"""
        from whoosh.query import Term, And, Or
        from whoosh.qparser import QueryParser
        
        with self.ix.searcher() as searcher:
            # 构建查询
            if query_string:
                # 多字段搜索
                parser = MultifieldParser(["title", "content", "description", "tags"], self.ix.schema)
                query = parser.parse(query_string)
            else:
                query = Every()
            
            # 添加过滤条件
            filters = []
            if category and category != 'all':
                filters.append(Term("category", category.lower()))
            
            if tags:
                for tag in tags:
                    filters.append(Term("tags", tag.lower()))
            
            if filters:
                query = And([query] + filters)
            
            results = searcher.search(query, limit=limit)
            
            # 格式化结果
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result['id'],
                    'title': result['title'],
                    'description': result['description'],
                    'tags': result['tags'].split(',') if result['tags'] else [],
                    'category': result['category'],
                    'file_type': result['file_type'],
                    'file_path': result['file_path'],
                    'file_size': result['file_size'],
                    'created_at': result['created_at'],
                    'updated_at': result['updated_at'],
                    'author': result['author'],
                    'language': result['language'],
                    'score': result.score
                })
            
            return {
                'results': formatted_results,
                'total': len(results),
                'query': query_string
            }
    
    def delete_document(self, doc_id):
        """从索引中删除文档"""
        writer = self.ix.writer()
        writer.delete_by_term('id', doc_id)
        writer.commit()
    
    def get_statistics(self):
        """获取搜索统计信息"""
        with self.ix.searcher() as searcher:
            return {
                'total_documents': searcher.doc_count(),
                'index_size': sum(f.stat().st_size for f in self.index_dir.iterdir() if f.is_file())
            }