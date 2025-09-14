#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库导入系统路由模块
提供文件上传、批量导入、知识管理等功能
"""

import os
import json
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import threading
import time

# 创建蓝图
import_routes = Blueprint('import_routes', __name__)

# 配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {
    'txt', 'md', 'csv', 'json', 'xml', 'yaml', 'yml',
    'pdf', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx',
    'jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'webp',
    'mp3', 'wav', 'flac', 'aac', 'ogg',
    'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm',
    'py', 'js', 'java', 'cpp', 'c', 'h', 'hpp', 'cs', 'php', 'rb', 'go', 'rs'
}

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'text'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'document'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'image'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'audio'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'video'), exist_ok=True)

# 全局变量
knowledge_base = {}
upload_tasks = {}

# 工具函数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_type(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in {'txt', 'md', 'csv', 'json', 'xml', 'yaml', 'yml'}:
        return 'text'
    elif ext in {'pdf', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx'}:
        return 'document'
    elif ext in {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'webp'}:
        return 'image'
    elif ext in {'mp3', 'wav', 'flac', 'aac', 'ogg'}:
        return 'audio'
    elif ext in {'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm'}:
        return 'video'
    elif ext in {'py', 'js', 'java', 'cpp', 'c', 'h', 'hpp', 'cs', 'php', 'rb', 'go', 'rs'}:
        return 'code'
    else:
        return 'other'

def process_file_content(file_path):
    """处理文件内容"""
    try:
        ext = file_path.rsplit('.', 1)[1].lower()
        
        if ext in {'txt', 'md', 'csv', 'json', 'xml', 'yaml', 'yml', 'py', 'js', 'java', 'cpp', 'c', 'h', 'hpp', 'cs', 'php', 'rb', 'go', 'rs'}:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext in {'pdf'}:
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text()
                    return text
            except ImportError:
                return "PDF文件，需要PyPDF2库来提取文本内容"
        elif ext in {'docx'}:
            try:
                from docx import Document
                doc = Document(file_path)
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            except ImportError:
                return "Word文档，需要python-docx库来提取文本内容"
        else:
            return f"{ext.upper()}文件，内容已保存"
    except Exception as e:
        return f"处理文件时出错: {str(e)}"

def save_knowledge_base():
    """保存知识库到文件"""
    try:
        with open('knowledge_base.json', 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存知识库失败: {e}")

def load_knowledge_base():
    """从文件加载知识库"""
    global knowledge_base
    try:
        if os.path.exists('knowledge_base.json'):
            with open('knowledge_base.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 确保加载的是字典格式
                if isinstance(data, dict):
                    knowledge_base = data
                else:
                    knowledge_base = {}
                    print("警告：知识库数据格式不正确，已重置为空")
        else:
            knowledge_base = {}
            print("知识库文件不存在，创建新的空知识库")
    except Exception as e:
        print(f"加载知识库失败: {e}")
        knowledge_base = {}

# 路由定义
@import_routes.route('/')
def index():
    """知识库导入主页面"""
    return render_template('knowledge_import.html')

@import_routes.route('/api/upload', methods=['POST'])
def upload_file():
    """上传单个文件"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "没有文件", "success": False})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "没有选择文件", "success": False})
        
        if not allowed_file(file.filename):
            return jsonify({"error": "文件类型不支持", "success": False})
        
        category = request.form.get('category', '未分类')
        
        # 保存文件
        filename = secure_filename(file.filename)
        file_type = get_file_type(filename)
        
        # 按类型保存到对应目录
        file_dir = os.path.join(UPLOAD_FOLDER, file_type)
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, filename)
        
        # 如果文件已存在，添加时间戳
        if os.path.exists(file_path):
            timestamp = str(int(time.time()))
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"
            file_path = os.path.join(file_dir, filename)
        
        file.save(file_path)
        
        # 处理文件内容
        content = process_file_content(file_path)
        
        # 生成知识ID
        knowledge_id = str(uuid.uuid4())
        
        # 创建知识条目
        knowledge_data = {
            "id": knowledge_id,
            "title": filename,
            "content": content,
            "category": category,
            "file_type": file_type,
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "content_length": len(content),
            "upload_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "original_name": file.filename
        }
        
        knowledge_base[knowledge_id] = knowledge_data
        save_knowledge_base()
        
        return jsonify({
            "success": True,
            "knowledge_id": knowledge_id,
            "filename": filename,
            "file_size": os.path.getsize(file_path),
            "content_length": len(content),
            "file_type": file_type,
            "category": category
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False})

@import_routes.route('/api/batch_upload', methods=['POST'])
def batch_upload():
    """批量上传文件"""
    try:
        results = []
        
        if 'files' not in request.files:
            return jsonify({"error": "没有文件", "success": False})
        
        files = request.files.getlist('files')
        category = request.form.get('category', '未分类')
        
        for file in files:
            if file and file.filename != '':
                try:
                    if not allowed_file(file.filename):
                        results.append({
                            "filename": file.filename,
                            "success": False,
                            "error": "文件类型不支持"
                        })
                        continue
                    
                    filename = secure_filename(file.filename)
                    file_type = get_file_type(filename)
                    
                    file_dir = os.path.join(UPLOAD_FOLDER, file_type)
                    os.makedirs(file_dir, exist_ok=True)
                    file_path = os.path.join(file_dir, filename)
                    
                    if os.path.exists(file_path):
                        timestamp = str(int(time.time()))
                        name, ext = os.path.splitext(filename)
                        filename = f"{name}_{timestamp}{ext}"
                        file_path = os.path.join(file_dir, filename)
                    
                    file.save(file_path)
                    content = process_file_content(file_path)
                    knowledge_id = str(uuid.uuid4())
                    
                    knowledge_data = {
                        "id": knowledge_id,
                        "title": filename,
                        "content": content,
                        "category": category,
                        "file_type": file_type,
                        "file_path": file_path,
                        "file_size": os.path.getsize(file_path),
                        "content_length": len(content),
                        "upload_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "original_name": file.filename
                    }
                    
                    knowledge_base[knowledge_id] = knowledge_data
                    
                    results.append({
                        "filename": file.filename,
                        "success": True,
                        "knowledge_id": knowledge_id,
                        "file_size": os.path.getsize(file_path),
                        "content_length": len(content),
                        "file_type": file_type
                    })
                    
                except Exception as e:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": str(e)
                    })
        
        save_knowledge_base()
        
        success_count = len([r for r in results if r["success"]])
        error_count = len([r for r in results if not r["success"]])
        
        return jsonify({
            "success": True,
            "results": results,
            "total": len(files),
            "success_count": success_count,
            "error_count": error_count
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False})

@import_routes.route('/api/statistics')
def get_statistics():
    """获取统计信息"""
    global knowledge_base
    try:
        # 确保knowledge_base是字典
        if not isinstance(knowledge_base, dict):
            knowledge_base = {}
            
        total_files = len(knowledge_base)
        total_size = 0
        file_types = {}
        categories = {}
        last_update = ''
        
        # 安全地遍历知识库
        for key, item in knowledge_base.items():
            if isinstance(item, dict):
                # 获取文件大小
                file_size = item.get('file_size', 0)
                if isinstance(file_size, (int, float)):
                    total_size += file_size
                
                # 获取文件类型
                file_type = str(item.get('file_type', 'unknown'))
                file_types[file_type] = file_types.get(file_type, 0) + 1
                
                # 获取分类
                category = str(item.get('category', '未分类'))
                categories[category] = categories.get(category, 0) + 1
                
                # 获取最后更新时间
                upload_time = str(item.get('upload_time', ''))
                if upload_time and upload_time > last_update:
                    last_update = upload_time
        
        return jsonify({
            "success": True,
            "data": {
                "total_knowledge": total_files,
                "total_size": total_size,
                "file_types": file_types,
                "categories": categories,
                "last_update": last_update
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False, 
            "error": str(e),
            "data": {
                "total_knowledge": 0,
                "total_size": 0,
                "file_types": {},
                "categories": {},
                "last_update": ""
            }
        })

@import_routes.route('/api/knowledge_list')
def get_knowledge_list():
    """获取知识列表"""
    global knowledge_base
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        category = request.args.get('category', '')
        file_type = request.args.get('file_type', '')
        
        # 确保knowledge_base是字典
        if not isinstance(knowledge_base, dict):
            knowledge_base = {}
        
        items = []
        for key, item in knowledge_base.items():
            if isinstance(item, dict):
                items.append(item)
        
        # 过滤
        if category:
            items = [item for item in items if str(item.get('category', '')) == str(category)]
        if file_type:
            items = [item for item in items if str(item.get('file_type', '')) == str(file_type)]
        
        # 排序
        items.sort(key=lambda x: str(x.get('upload_time', '')), reverse=True)
        
        # 分页
        start = (page - 1) * per_page
        end = start + per_page
        paginated_items = items[start:end]
        
        return jsonify({
            "success": True,
            "data": {
                "items": paginated_items,
                "total": len(items),
                "page": page,
                "per_page": per_page,
                "pages": (len(items) + per_page - 1) // per_page
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "data": {
                "items": [],
                "total": 0,
                "page": 1,
                "per_page": per_page,
                "pages": 0
            }
        })

@import_routes.route('/api/export_knowledge')
def export_knowledge():
    """导出知识库"""
    try:
        format_type = request.args.get('format', 'json')
        
        if format_type == 'json':
            export_data = {
                "export_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_items": len(knowledge_base),
                "knowledge_base": knowledge_base
            }
            
            filename = f"knowledge_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join('exports', filename)
            os.makedirs('exports', exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return send_file(filepath, as_attachment=True, download_name=filename)
        
        else:
            return jsonify({"error": "不支持的导出格式", "success": False})
            
    except Exception as e:
        return jsonify({"error": str(e), "success": False})

@import_routes.route('/api/delete_knowledge/<knowledge_id>', methods=['DELETE'])
def delete_knowledge(knowledge_id):
    """删除知识条目"""
    try:
        if knowledge_id in knowledge_base:
            # 删除文件
            file_path = knowledge_base[knowledge_id].get('file_path')
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"删除文件失败: {e}")
            
            # 删除知识条目
            del knowledge_base[knowledge_id]
            save_knowledge_base()
            
            return jsonify({"success": True, "message": "删除成功"})
        else:
            return jsonify({"error": "知识条目不存在", "success": False})
            
    except Exception as e:
        return jsonify({"error": str(e), "success": False})

# 创建Flask应用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局变量
knowledge_base = {}
lock = threading.Lock()

# 知识库文件路径
KNOWLEDGE_FILE = 'knowledge_base.json'

# 在应用启动时加载知识库
try:
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
    else:
        knowledge_base = {}
except Exception as e:
    print(f"加载知识库文件失败: {e}")
    knowledge_base = {}

def maintenance_task():
    """后台维护任务"""
    while True:
        try:
            time.sleep(300)  # 每5分钟执行一次
            save_knowledge_base()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 知识库自动保存完成")
        except Exception as e:
            print(f"维护任务失败: {e}")

maintenance_thread = threading.Thread(target=maintenance_task, daemon=True)
maintenance_thread.start()

@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    """与A管理模型进行对话"""
    global knowledge_base
    
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': '消息不能为空'
            }), 400
            
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({
                'success': False,
                'error': '消息不能为空'
            }), 400

        # 这里可以集成真实的AI模型API
        # 目前使用智能回复系统
        response = generate_ai_response(user_message, knowledge_base)
        
        return jsonify({
            'success': True,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'AI对话错误: {str(e)}'
        }), 500

def generate_ai_response(message, knowledge_data):
    """生成AI智能回复"""
    message_lower = message.lower()
    
    # 项目分析相关
    if any(word in message_lower for word in ['项目', '状态', '分析', '状况']):
        return analyze_project_status(knowledge_data)
    
    # 系统优化相关
    elif any(word in message_lower for word in ['优化', '建议', '改进']):
        return provide_optimization_suggestions(knowledge_data)
    
    # 训练计划相关
    elif any(word in message_lower for word in ['训练', '计划', '学习']):
        return generate_training_plan(knowledge_data)
    
    # 知识库查询
    elif any(word in message_lower for word in ['知识', '查询', '搜索', '查找']):
        return search_knowledge_base(message, knowledge_data)
    
    # 系统信息
    elif any(word in message_lower for word in ['系统', '信息', '配置']):
        return get_system_info()
    
    # 默认回复
    else:
        return get_general_response(message)

def analyze_project_status(knowledge_data):
    """分析项目状态"""
    total_files = len(knowledge_data) if isinstance(knowledge_data, dict) else 0
    categories = set()
    file_types = set()
    
    if isinstance(knowledge_data, dict):
        for key, item in knowledge_data.items():
            if isinstance(item, dict):
                categories.add(item.get('category', '未分类'))
                file_type = item.get('file_type', 'unknown')
                if file_type != 'unknown':
                    file_types.add(file_type)
    
    return f"""📊 项目状态分析报告

当前知识库状态：
• 总文件数：{total_files}个
• 分类数量：{len(categories)}个
• 文件类型：{', '.join(file_types) if file_types else '暂无'}

系统运行正常，所有模块都在稳定工作。建议定期更新知识库以保持数据新鲜度。"""

def provide_optimization_suggestions(knowledge_data):
    """提供优化建议"""
    suggestions = [
        "建议增加更多多样化的训练数据",
        "可以优化知识分类体系",
        "考虑增加实时数据更新机制",
        "建议完善文件元数据信息",
        "可以添加自动标签生成功能"
    ]
    
    return f"""💡 优化建议

基于当前系统分析，我建议：
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(suggestions))}

这些优化将有助于提升系统性能和用户体验。"""

def generate_training_plan(knowledge_data):
    """生成训练计划"""
    return """🎯 训练计划建议

阶段1：数据准备（1-2天）
• 收集和整理训练数据
• 验证数据质量和完整性

阶段2：模型训练（3-5天）
• 配置训练参数
• 监控训练进度和性能指标

阶段3：评估优化（1-2天）
• 测试模型效果
• 根据结果调整优化

阶段4：部署上线（1天）
• 部署到生产环境
• 设置监控和告警

预计总用时：6-10天"""

def search_knowledge_base(query, knowledge_data):
    """搜索知识库"""
    if not isinstance(knowledge_data, dict):
        return "知识库当前为空，建议先导入一些数据。"
    
    keywords = query.lower().split()
    matches = []
    
    for key, item in knowledge_data.items():
        if isinstance(item, dict):
            content = str(item).lower()
            if any(keyword in content for keyword in keywords):
                matches.append(item.get('title', key))
    
    if matches:
        return f"找到 {len(matches)} 个相关结果：\n{chr(10).join(f'• {m}' for m in matches[:5])}"
    else:
        return "未找到相关内容，建议尝试其他关键词。"

def get_system_info():
    """获取系统信息"""
    return """🖥️ 系统信息

• 运行环境：Python Flask
• 知识库路径：当前工作目录
• API端口：8003
• 支持功能：文件上传、批量导入、实时对话
• 系统状态：运行正常

可以通过 /api/statistics 和 /api/knowledge_list 获取详细数据。"""

def get_general_response(message):
    """通用回复"""
    responses = [
        f"我理解您的问题：{message[:50]}... 让我为您提供一些见解：",
        "这是一个很好的问题！让我分析一下：",
        "基于A管理模型的理解，我认为：",
        "让我为您详细解释一下："
    ]
    
    return f"""{responses[hash(message) % len(responses)]}

作为您的AI助手，我可以帮助您：
• 分析项目状态和进展
• 提供系统优化建议
• 制定训练和学习计划
• 查询知识库信息
• 解答技术问题

请告诉我您具体想了解什么？"""

# 添加聊天页面路由
@app.route('/chat')
def chat_page():
    """AI对话页面"""
    return render_template('ai_chat.html')

# 在文件末尾添加健康检查路由
@app.route('/api/health')
def health_check():
    """系统健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime': 'running'
    })

if __name__ == '__main__':
    # 启动后台维护线程
    maintenance_thread = threading.Thread(target=maintenance_task, daemon=True)
    maintenance_thread.start()
    
    print("🚀 知识库导入系统启动成功！")
    print("📊 访问地址：http://localhost:8003")
    print("📁 导入页面：http://localhost:8003")
    print("🤖 AI对话：http://localhost:8003/chat")
    print("📈 API文档：http://localhost:8003/api/docs")
    print("📋 知识列表：http://localhost:8003/api/knowledge_list")
    print("📊 统计信息：http://localhost:8003/api/statistics")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5009, debug=True, use_reloader=False)