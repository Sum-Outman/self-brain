# Copyright 2025 The AI Management System Authors
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

# 知识库文件导入器
# Knowledge Base File Importer

import os
import json
import csv
import PyPDF2
import docx
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from .knowledge_model import KnowledgeExpert

class KnowledgeImporter:
    def __init__(self, knowledge_expert: KnowledgeExpert):
        """
        知识库文件导入器 | Knowledge Base File Importer
        
        参数 Parameters:
        knowledge_expert: 知识库专家实例 | Knowledge expert instance
        """
        self.knowledge_expert = knowledge_expert
        self.supported_formats = {
            'txt': self._import_txt,
            'json': self._import_json,
            'csv': self._import_csv,
            'pdf': self._import_pdf,
            'docx': self._import_docx,
            'md': self._import_markdown
        }
        
        # 知识领域映射
        self.domain_keywords = {
            'physics': ['物理', 'physics', '力学', '电磁学', '量子', '运动', '能量', '力'],
            'mathematics': ['数学', 'mathematics', '代数', '几何', '微积分', '方程', '统计', '概率'],
            'chemistry': ['化学', 'chemistry', '分子', '反应', '元素', '化合物', '化学式', '反应式'],
            'medicine': ['医学', 'medicine', '医疗', '疾病', '药物', '解剖', '生理', '病理'],
            'law': ['法学', 'law', '法律', '法规', '条约', '合同', '刑法', '民法'],
            'history': ['历史', 'history', '古代', '近代', '事件', '朝代', '文明', '文化'],
            'sociology': ['社会学', 'sociology', '社会', '文化', '群体', '组织', '制度', '变迁'],
            'humanities': ['人文学', 'humanities', '人文', '哲学', '文学', '艺术', '宗教', '伦理'],
            'psychology': ['心理学', 'psychology', '心理', '行为', '认知', '情绪', '人格', '治疗'],
            'economics': ['经济学', 'economics', '经济', '市场', '金融', '货币', '贸易', 'GDP'],
            'management': ['管理学', 'management', '管理', '组织', '战略', '领导', '决策', '效率'],
            'mechanical_engineering': ['机械工程', 'mechanical', '机械', '设计', '制造', '机构', '传动', '液压'],
            'electronic_engineering': ['电子工程', 'electronic', '电路', '信号', '系统', '半导体', '数字', '模拟'],
            'food_engineering': ['食品工程', 'food', '食品', '加工', '安全', '营养', '保鲜', '包装'],
            'chemical_engineering': ['化学工程', 'chemical', '化工', '反应器', '工艺', '分离', '传质', '传热']
        }
        
        logging.info("知识库文件导入器初始化完成 | Knowledge importer initialized")

    def import_file(self, file_path: str, category: Optional[str] = None, 
                   metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        导入知识文件 | Import knowledge file
        
        参数 Parameters:
        file_path: 文件路径 | File path
        category: 知识类别 | Knowledge category
        metadata: 元数据 | Metadata
        
        返回 Returns:
        导入结果 | Import result
        """
        try:
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'error': f"文件不存在: {file_path} | File not found: {file_path}",
                    'imported_count': 0
                }
            
            file_extension = Path(file_path).suffix.lower()[1:]  # 去掉点号
            
            if file_extension not in self.supported_formats:
                return {
                    'success': False,
                    'error': f"不支持的文件格式: {file_extension} | Unsupported file format: {file_extension}",
                    'imported_count': 0
                }
            
            # 自动检测类别
            if not category:
                category = self._detect_category(file_path)
            
            # 导入文件
            import_func = self.supported_formats[file_extension]
            result = import_func(file_path, category, metadata)
            
            return {
                'success': True,
                'imported_count': result.get('count', 0),
                'category': category,
                'file_type': file_extension,
                'details': result
            }
            
        except Exception as e:
            logging.error(f"文件导入错误: {str(e)} | File import error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'imported_count': 0
            }

    def _detect_category(self, file_path: str) -> str:
        """
        自动检测文件所属知识类别 | Auto-detect knowledge category from file
        
        参数 Parameters:
        file_path: 文件路径 | File path
        
        返回 Returns:
        检测到的类别 | Detected category
        """
        try:
            # 读取文件内容进行关键词分析
            content = self._read_file_content(file_path)
            if not content:
                return "general"
            
            # 计算每个类别的匹配分数
            category_scores = {}
            for category, keywords in self.domain_keywords.items():
                score = 0
                for keyword in keywords:
                    if keyword.lower() in content.lower():
                        score += 1
                category_scores[category] = score
            
            # 返回得分最高的类别
            best_category = max(category_scores.items(), key=lambda x: x[1])
            if best_category[1] > 0:
                return best_category[0]
            else:
                return "general"
                
        except Exception as e:
            logging.warning(f"类别检测失败: {str(e)} | Category detection failed: {str(e)}")
            return "general"

    def _read_file_content(self, file_path: str, max_length: int = 10000) -> str:
        """
        读取文件内容 | Read file content
        
        参数 Parameters:
        file_path: 文件路径 | File path
        max_length: 最大读取长度 | Maximum read length
        
        返回 Returns:
        文件内容 | File content
        """
        try:
            file_extension = Path(file_path).suffix.lower()[1:]
            
            if file_extension == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read(max_length)
            elif file_extension == 'pdf':
                return self._extract_pdf_text(file_path, max_length)
            elif file_extension == 'docx':
                return self._extract_docx_text(file_path, max_length)
            else:
                # 对于其他格式，尝试以文本方式读取
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read(max_length)
                except:
                    return ""
                    
        except Exception as e:
            logging.error(f"读取文件内容错误: {str(e)} | Read file content error: {str(e)}")
            return ""

    def _import_txt(self, file_path: str, category: str, metadata: Optional[Dict]) -> Dict:
        """
        导入文本文件 | Import text file
        """
        imported_count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 将文本分割成段落
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 50:  # 只处理有实质内容的段落
                    question = f"关于{category}的知识段落{i+1}"
                    answer = paragraph
                    
                    self.knowledge_expert.add_knowledge(
                        category=category,
                        question=question,
                        answer=answer,
                        source=metadata.get('source', file_path) if metadata else file_path
                    )
                    imported_count += 1
                    
            return {'count': imported_count, 'paragraphs': len(paragraphs)}
            
        except Exception as e:
            logging.error(f"导入文本文件错误: {str(e)} | Import text file error: {str(e)}")
            return {'count': 0, 'error': str(e)}

    def _import_json(self, file_path: str, category: str, metadata: Optional[Dict]) -> Dict:
        """
        导入JSON文件 | Import JSON file
        """
        imported_count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'question' in item and 'answer' in item:
                        self.knowledge_expert.add_knowledge(
                            category=category,
                            question=item['question'],
                            answer=item['answer'],
                            subcategory=item.get('subcategory'),
                            source=item.get('source', metadata.get('source', file_path) if metadata else file_path)
                        )
                        imported_count += 1
            elif isinstance(data, dict):
                # 处理字典格式的知识
                for key, value in data.items():
                    if isinstance(value, str):
                        self.knowledge_expert.add_knowledge(
                            category=category,
                            question=key,
                            answer=value,
                            source=metadata.get('source', file_path) if metadata else file_path
                        )
                        imported_count += 1
                        
            return {'count': imported_count, 'items_processed': len(data) if isinstance(data, list) else len(data)}
            
        except Exception as e:
            logging.error(f"导入JSON文件错误: {str(e)} | Import JSON file error: {str(e)}")
            return {'count': 0, 'error': str(e)}

    def _import_csv(self, file_path: str, category: str, metadata: Optional[Dict]) -> Dict:
        """
        导入CSV文件 | Import CSV file
        """
        imported_count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'question' in row and 'answer' in row:
                        self.knowledge_expert.add_knowledge(
                            category=category,
                            question=row['question'],
                            answer=row['answer'],
                            subcategory=row.get('subcategory'),
                            source=row.get('source', metadata.get('source', file_path) if metadata else file_path)
                        )
                        imported_count += 1
                        
            return {'count': imported_count}
            
        except Exception as e:
            logging.error(f"导入CSV文件错误: {str(e)} | Import CSV file error: {str(e)}")
            return {'count': 0, 'error': str(e)}

    def _import_pdf(self, file_path: str, category: str, metadata: Optional[Dict]) -> Dict:
        """
        导入PDF文件 | Import PDF file
        """
        imported_count = 0
        try:
            text = self._extract_pdf_text(file_path)
            if not text:
                return {'count': 0, 'error': '无法提取PDF文本'}
                
            # 将文本分割成有意义的段落
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 100]
            
            for i, paragraph in enumerate(paragraphs):
                question = f"PDF文档第{i+1}段关于{category}"
                answer = paragraph
                
                self.knowledge_expert.add_knowledge(
                    category=category,
                    question=question,
                    answer=answer,
                    source=metadata.get('source', file_path) if metadata else file_path
                )
                imported_count += 1
                
            return {'count': imported_count, 'paragraphs': len(paragraphs)}
            
        except Exception as e:
            logging.error(f"导入PDF文件错误: {str(e)} | Import PDF file error: {str(e)}")
            return {'count': 0, 'error': str(e)}

    def _extract_pdf_text(self, file_path: str, max_length: int = 50000) -> str:
        """
        提取PDF文本 | Extract PDF text
        """
        try:
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    if len(text) >= max_length:
                        break
            return text[:max_length]
        except Exception as e:
            logging.error(f"提取PDF文本错误: {str(e)} | Extract PDF text error: {str(e)}")
            return ""

    def _import_docx(self, file_path: str, category: str, metadata: Optional[Dict]) -> Dict:
        """
        导入DOCX文件 | Import DOCX file
        """
        imported_count = 0
        try:
            text = self._extract_docx_text(file_path)
            if not text:
                return {'count': 0, 'error': '无法提取DOCX文本'}
                
            # 将文本分割成有意义的段落
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 100]
            
            for i, paragraph in enumerate(paragraphs):
                question = f"文档第{i+1}段关于{category}"
                answer = paragraph
                
                self.knowledge_expert.add_knowledge(
                    category=category,
                    question=question,
                    answer=answer,
                    source=metadata.get('source', file_path) if metadata else file_path
                )
                imported_count += 1
                
            return {'count': imported_count, 'paragraphs': len(paragraphs)}
            
        except Exception as e:
            logging.error(f"导入DOCX文件错误: {str(e)} | Import DOCX file error: {str(e)}")
            return {'count': 0, 'error': str(e)}

    def _extract_docx_text(self, file_path: str, max_length: int = 50000) -> str:
        """
        提取DOCX文本 | Extract DOCX text
        """
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
                if len(text) >= max_length:
                    break
            return text[:max_length]
        except Exception as e:
            logging.error(f"提取DOCX文本错误: {str(e)} | Extract DOCX text error: {str(e)}")
            return ""

    def _import_markdown(self, file_path: str, category: str, metadata: Optional[Dict]) -> Dict:
        """
        导入Markdown文件 | Import Markdown file
        """
        imported_count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 分割Markdown章节
            sections = []
            current_section = []
            lines = content.split('\n')
            
            for line in lines:
                if line.startswith('#') and current_section:
                    sections.append('\n'.join(current_section))
                    current_section = [line]
                else:
                    current_section.append(line)
                    
            if current_section:
                sections.append('\n'.join(current_section))
                
            for i, section in enumerate(sections):
                if len(section.strip()) > 100:
                    # 提取标题作为问题
                    lines = section.split('\n')
                    title = next((line for line in lines if line.startswith('#')), f"Markdown章节{i+1}")
                    title = title.lstrip('#').strip()
                    
                    question = f"{title} - {category}"
                    answer = section
                    
                    self.knowledge_expert.add_knowledge(
                        category=category,
                        question=question,
                        answer=answer,
                        source=metadata.get('source', file_path) if metadata else file_path
                    )
                    imported_count += 1
                    
            return {'count': imported_count, 'sections': len(sections)}
            
        except Exception as e:
            logging.error(f"导入Markdown文件错误: {str(e)} | Import Markdown file error: {str(e)}")
            return {'count': 0, 'error': str(e)}

    def batch_import(self, directory_path: str, file_pattern: str = "*", 
                    category: Optional[str] = None) -> Dict[str, Any]:
        """
        批量导入目录中的文件 | Batch import files from directory
        
        参数 Parameters:
        directory_path: 目录路径 | Directory path
        file_pattern: 文件模式 | File pattern
        category: 知识类别 | Knowledge category
        
        返回 Returns:
        批量导入结果 | Batch import results
        """
        results = {
            'total_files': 0,
            'successful_imports': 0,
            'failed_imports': 0,
            'total_knowledge_items': 0,
            'file_results': []
        }
        
        try:
            if not os.path.exists(directory_path):
                return {
                    'success': False,
                    'error': f"目录不存在: {directory_path} | Directory not found: {directory_path}"
                }
                
            # 查找匹配的文件
            import glob
            files = glob.glob(os.path.join(directory_path, file_pattern))
            
            for file_path in files:
                if os.path.isfile(file_path):
                    results['total_files'] += 1
                    file_result = self.import_file(file_path, category)
                    
                    if file_result['success']:
                        results['successful_imports'] += 1
                        results['total_knowledge_items'] += file_result['imported_count']
                    else:
                        results['failed_imports'] += 1
                        
                    results['file_results'].append({
                        'file_path': file_path,
                        'success': file_result['success'],
                        'imported_count': file_result['imported_count'],
                        'error': file_result.get('error'),
                        'category': file_result.get('category')
                    })
                    
            results['success'] = True
            return results
            
        except Exception as e:
            logging.error(f"批量导入错误: {str(e)} | Batch import error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'total_files': results['total_files'],
                'successful_imports': results['successful_imports'],
                'failed_imports': results['failed_imports']
            }

# 测试代码
if __name__ == '__main__':
    # 测试知识库文件导入器
    # Test knowledge file importer
    logging.basicConfig(level=logging.INFO)
    
    # 创建知识库专家实例
    expert = KnowledgeExpert()
    
    # 创建导入器
    importer = KnowledgeImporter(expert)
    
    # 测试单个文件导入
    test_file = "test_knowledge.json"
    if os.path.exists(test_file):
        result = importer.import_file(test_file, "physics")
        print("单个文件导入结果 | Single file import result:", result)
    
    # 测试批量导入
    test_dir = "knowledge_docs"
    if os.path.exists(test_dir):
        batch_result = importer.batch_import(test_dir, "*.txt", "general")
        print("批量导入结果 | Batch import result:", batch_result)
    
    expert.close()