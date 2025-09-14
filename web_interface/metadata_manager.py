import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
import mimetypes

class KnowledgeMetadataManager:
    def __init__(self, storage_path):
        self.storage_path = Path(storage_path)
        self.metadata_dir = self.storage_path / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
    
    def create_metadata(self, file_path, title=None, description=None, tags=None, category=None, author=None):
        """为知识条目创建元数据"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # 生成唯一ID
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:16]
        
        # 自动提取文件信息
        stat = file_path.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        metadata = {
            "id": file_hash,
            "title": title or file_path.stem,
            "description": description or f"Knowledge entry: {file_path.name}",
            "tags": tags or [category] if category else ["uncategorized"],
            "category": category or self._auto_classify(file_path),
            "author": author or "system",
            "file_path": str(file_path.relative_to(self.storage_path)),
            "file_name": file_path.name,
            "file_size": stat.st_size,
            "file_type": mime_type or "unknown",
            "file_extension": file_path.suffix.lower(),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "last_accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
            "version": "1.0",
            "checksum": self._calculate_checksum(file_path),
            "content_preview": self._extract_preview(file_path),
            "language": self._detect_language(file_path),
            "indexed": False,
            "processed": False
        }
        
        # 保存元数据
        metadata_file = self.metadata_dir / f"{file_hash}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata
    
    def _auto_classify(self, file_path):
        """根据文件类型自动分类"""
        extension = file_path.suffix.lower()
        classification_map = {
            '.txt': 'text',
            '.md': 'documentation',
            '.py': 'code',
            '.js': 'code',
            '.json': 'config',
            '.yaml': 'config',
            '.yml': 'config',
            '.csv': 'data',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.png': 'image',
            '.gif': 'image',
            '.mp3': 'audio',
            '.wav': 'audio',
            '.mp4': 'video',
            '.avi': 'video',
            '.pdf': 'document'
        }
        return classification_map.get(extension, 'other')
    
    def _calculate_checksum(self, file_path):
        """计算文件校验和"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
    
    def _extract_preview(self, file_path, max_length=500):
        """提取文件内容预览"""
        try:
            if file_path.suffix.lower() in ['.txt', '.md', '.py', '.js', '.json']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(max_length * 2)
                    if len(content) > max_length:
                        content = content[:max_length] + "..."
                    return content
            return f"Binary file: {file_path.suffix}"
        except:
            return "Unable to extract preview"
    
    def _detect_language(self, file_path):
        """检测文件语言"""
        try:
            if file_path.suffix.lower() in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)
                    # 简单的语言检测
                    chinese_chars = len([c for c in content if '\u4e00' <= c <= '\u9fff'])
                    if chinese_chars > len(content) * 0.1:
                        return "zh"
                    return "en"
            return "unknown"
        except:
            return "unknown"
    
    def get_metadata(self, metadata_id):
        """获取元数据"""
        metadata_file = self.metadata_dir / f"{metadata_id}.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def update_metadata(self, metadata_id, updates):
        """更新元数据"""
        metadata = self.get_metadata(metadata_id)
        if metadata:
            metadata.update(updates)
            metadata['updated_at'] = datetime.now().isoformat()
            metadata_file = self.metadata_dir / f"{metadata_id}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return metadata
        return None
    
    def search_metadata(self, query, category=None, tags=None):
        """搜索元数据"""
        results = []
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                    # 全文搜索
                    match = False
                    if query:
                        query = query.lower()
                        if query in metadata.get('title', '').lower():
                            match = True
                        if query in metadata.get('description', '').lower():
                            match = True
                        if query in str(metadata.get('tags', [])).lower():
                            match = True
                        if query in metadata.get('content_preview', '').lower():
                            match = True
                    else:
                        match = True
                    
                    # 分类过滤
                    if category and category != 'all' and metadata.get('category') != category:
                        match = False
                    
                    # 标签过滤
                    if tags:
                        entry_tags = set(metadata.get('tags', []))
                        if not any(tag in entry_tags for tag in tags):
                            match = False
                    
                    if match:
                        results.append(metadata)
            except Exception as e:
                print(f"Error reading metadata file {metadata_file}: {e}")
        
        return sorted(results, key=lambda x: x.get('updated_at', ''), reverse=True)
    
    def get_all_metadata(self):
        """获取所有元数据"""
        results = []
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    results.append(json.load(f))
            except:
                continue
        return results