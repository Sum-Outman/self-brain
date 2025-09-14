import os
import json
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, KEYWORD, ID, DATETIME, NUMERIC
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import Every, Term, And, Or
from whoosh.analysis import StemmingAnalyzer
import jieba
import re
from pathlib import Path

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
        """创建或打开索引"""
        try:
            self.ix = open_dir(str(self.index_dir))
        except:
            self.ix = create_in(str(self.index_dir), self.schema)
    
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