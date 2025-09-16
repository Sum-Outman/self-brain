#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库导入系统主应用
集成了文件上传、知识管理和AI对话功能
"""

from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import os
import json
import uuid
import hashlib
import threading
import time
from datetime import datetime

# 创建Flask应用
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# 确保必要的目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'text'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'document'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'image'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'audio'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'video'), exist_ok=True)
os.makedirs('exports', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# 全局变量
knowledge_base = {}
upload_tasks = {}
lock = threading.Lock()

# 知识库文件路径
KNOWLEDGE_FILE = 'knowledge_base.json'

# 支持的文件类型
ALLOWED_EXTENSIONS = {
    'txt', 'md', 'csv', 'json', 'xml', 'yaml', 'yml',
    'pdf', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx',
    'jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'webp',
    'mp3', 'wav', 'flac', 'aac', 'ogg',
    'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm',
    'py', 'js', 'java', 'cpp', 'c', 'h', 'hpp', 'cs', 'php', 'rb', 'go', 'rs'
}

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
        with open(KNOWLEDGE_FILE, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存知识库失败: {e}")

def load_knowledge_base():
    """从文件加载知识库"""
    global knowledge_base
    try:
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    knowledge_base = data
                else:
                    knowledge_base = {}
        else:
            knowledge_base = {}
    except Exception as e:
        print(f"加载知识库失败: {e}")
        knowledge_base = {}

# 在应用启动时加载知识库
load_knowledge_base()

# 路由定义
@app.route('/')
def index():
    """原始主页 - 使用以前的模板"""
    return render_template('original_index.html')

@app.route('/dashboard')
def dashboard():
    """重定向到统一控制面板"""
    return redirect(url_for('index'))

@app.route('/knowledge')
def knowledge_page():
    """知识库管理页面"""
    return render_template('knowledge_manage.html')

@app.route('/import')
def import_page():
    """知识库导入页面"""
    return render_template('knowledge_import.html')

@app.route('/chat')
def chat_page():
    """AI对话页面"""
    return render_template('ai_chat.html')

@app.route('/analytics')
def analytics_page():
    """数据分析页面"""
    return render_template('analytics.html')

@app.route('/settings')
@app.route('/settings/<path:page>')
def settings_page(page='general'):
    """系统设置页面"""
    valid_pages = ['general', 'ai', 'storage', 'security', 'notifications', 'api']
    if page not in valid_pages:
        page = 'general'
    return render_template('settings.html', current_page=page)

@app.route('/help')
def help_page():
    """帮助文档页面"""
    return render_template('help.html')

@app.route('/api/upload', methods=['POST'])
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
        file_dir = os.path.join(app.config['UPLOAD_FOLDER'], file_type)
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

@app.route('/api/batch_upload', methods=['POST'])
def batch_upload():
    """批量上传文件"""
    try:
        if 'files' not in request.files:
            return jsonify({"error": "没有文件", "success": False})
        
        files = request.files.getlist('files')
        category = request.form.get('category', '未分类')
        
        results = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_type = get_file_type(filename)
                
                file_dir = os.path.join(app.config['UPLOAD_FOLDER'], file_type)
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
                    "filename": filename,
                    "success": True,
                    "knowledge_id": knowledge_id
                })
            else:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "文件类型不支持"
                })
        
        save_knowledge_base()
        return jsonify({"success": True, "results": results})
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False})

@app.route('/api/knowledge_list')
def get_knowledge_list():
    """获取知识列表"""
    global knowledge_base
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        search = request.args.get('search', '').lower()
        category = request.args.get('category', '')
        file_type = request.args.get('file_type', '')
        
        if not isinstance(knowledge_base, dict):
            knowledge_base = {}
        
        items = []
        for key, item in knowledge_base.items():
            if isinstance(item, dict):
                match = True
                
                if search and search not in str(item).lower():
                    match = False
                
                if category and item.get('category', '') != category:
                    match = False
                
                if file_type and item.get('file_type', '') != file_type:
                    match = False
                
                if match:
                    items.append(item)
        
        items.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
        
        total = len(items)
        start = (page - 1) * per_page
        end = start + per_page
        paginated_items = items[start:end]
        
        pages = (total + per_page - 1) // per_page
        
        return jsonify({
            'success': True,
            'data': {
                'items': paginated_items,
                'total': total,
                'page': page,
                'per_page': per_page,
                'pages': pages
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {
                'items': [],
                'total': 0,
                'page': 1,
                'per_page': 20,
                'pages': 1
            }
        })

@app.route('/api/statistics')
def get_statistics():
    """获取统计信息"""
    global knowledge_base
    try:
        if not isinstance(knowledge_base, dict):
            knowledge_base = {}
        
        total_knowledge = len(knowledge_base)
        total_size = 0
        categories = {}
        file_types = {}
        last_update = ""
        
        for key, item in knowledge_base.items():
            if isinstance(item, dict):
                total_size += item.get('file_size', 0)
                
                category = item.get('category', '未分类')
                categories[category] = categories.get(category, 0) + 1
                
                file_type = item.get('file_type', 'unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1
                
                upload_time = item.get('upload_time', '')
                if upload_time > last_update:
                    last_update = upload_time
        
        return jsonify({
            'success': True,
            'data': {
                'total_knowledge': total_knowledge,
                'total_size': total_size,
                'categories': categories,
                'file_types': file_types,
                'last_update': last_update
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {
                'total_knowledge': 0,
                'total_size': 0,
                'categories': {},
                'file_types': {},
                'last_update': ''
            }
        })

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
    categories = {}
    file_types = {}
    
    if isinstance(knowledge_data, dict):
        for key, item in knowledge_data.items():
            if isinstance(item, dict):
                category = item.get('category', '未分类')
                categories[category] = categories.get(category, 0) + 1
                
                file_type = item.get('file_type', 'unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1
    
    return f"""📊 项目状态分析报告

当前知识库状态：
• 总文件数：{total_files}个
• 分类统计：{', '.join(f'{k}:{v}' for k, v in categories.items()) if categories else '暂无分类'}
• 文件类型：{', '.join(f'{k}:{v}' for k, v in file_types.items()) if file_types else '暂无'}

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

@app.route('/api/health')
def health_check():
    """系统健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime': 'running'
    })

# 后台维护任务
def maintenance_task():
    """后台维护任务"""
    while True:
        try:
            save_knowledge_base()
            time.sleep(300)  # 每5分钟保存一次
        except Exception as e:
            print(f"后台维护任务出错: {e}")
            time.sleep(60)

# 启动后台维护线程
maintenance_thread = threading.Thread(target=maintenance_task, daemon=True)
maintenance_thread.start()

if __name__ == '__main__':
    print("🚀 知识库导入系统启动成功！")
    print("📊 访问地址：http://localhost:8003")
    print("📁 导入页面：http://localhost:8003")
    print("🤖 AI对话：http://localhost:8003/chat")
    print("📈 API文档：http://localhost:8003/api/health")
    print("📋 知识列表：http://localhost:8003/api/knowledge_list")
    print("📊 统计信息：http://localhost:8003/api/statistics")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=8003, debug=True, use_reloader=False)

# 在现有导入之后添加多媒体支持
from flask import Flask, request, jsonify, render_template, send_from_directory, session
from werkzeug.utils import secure_filename
import os
import json
import uuid
import hashlib
import threading
import time
from datetime import datetime
import base64
import io
import wave
import numpy as np
from PIL import Image
import cv2
import speech_recognition as sr
import pygame
import subprocess
import tempfile

# 添加新的配置
app.config['MAX_AUDIO_SIZE'] = 50 * 1024 * 1024  # 50MB
app.config['MAX_VIDEO_SIZE'] = 100 * 1024 * 1024  # 100MB
app.config['MAX_IMAGE_SIZE'] = 20 * 1024 * 1024    # 20MB

# 确保多媒体目录存在
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'chat_audio'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'chat_video'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'chat_images'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'chat_files'), exist_ok=True)

# 语音合成和识别配置
recognizer = sr.Recognizer()
pygame.mixer.init()

# 存储聊天记录
chat_sessions = {}

# 多媒体处理函数
def process_audio_file(audio_data, filename):
    """处理音频文件"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'chat_audio', filename)
        
        # 保存音频文件
        with open(file_path, 'wb') as f:
            f.write(audio_data)
        
        # 使用语音识别转换文本
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio, language='zh-CN')
                return {
                    'success': True,
                    'text': text,
                    'file_path': file_path,
                    'file_size': len(audio_data),
                    'duration': len(audio_data) / 44100  # 估算时长
                }
            except sr.UnknownValueError:
                return {
                    'success': False,
                    'error': '无法识别音频内容',
                    'file_path': file_path
                }
            except sr.RequestError as e:
                return {
                    'success': False,
                    'error': f'语音识别服务错误: {e}',
                    'file_path': file_path
                }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def process_image_file(image_data, filename):
    """处理图片文件"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'chat_images', filename)
        
        # 保存图片
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        # 处理图片信息
        img = Image.open(io.BytesIO(image_data))
        
        return {
            'success': True,
            'file_path': file_path,
            'width': img.width,
            'height': img.height,
            'format': img.format,
            'size': len(image_data),
            'description': f'图片尺寸: {img.width}x{img.height}, 格式: {img.format}'
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def process_video_file(video_data, filename):
    """处理视频文件"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'chat_video', filename)
        
        # 保存视频文件
        with open(file_path, 'wb') as f:
            f.write(video_data)
        
        # 使用OpenCV获取视频信息
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        
        return {
            'success': True,
            'file_path': file_path,
            'duration': duration,
            'width': int(width),
            'height': int(height),
            'fps': fps,
            'size': len(video_data)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def text_to_speech(text, session_id):
    """文本转语音"""
    try:
        # 使用系统TTS或简单实现
        temp_file = os.path.join(tempfile.gettempdir(), f'tts_{session_id}.wav')
        
        # 这里可以使用更复杂的TTS库
        # 简单实现：使用系统命令
        if os.name == 'nt':  # Windows
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\"{text}\")"'
            subprocess.run(cmd, shell=True)
            return {'success': True, 'audio_url': None, 'message': '语音播放成功'}
        else:
            return {'success': False, 'error': '语音合成功能暂不支持当前系统'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# 多媒体聊天API
@app.route('/api/chat/upload', methods=['POST'])
def upload_chat_media():
    """上传聊天媒体文件"""
    try:
        media_type = request.form.get('type', 'text')
        session_id = request.form.get('session_id', 'default')
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        filename = secure_filename(file.filename)
        file_data = file.read()
        
        result = None
        
        if media_type == 'audio':
            if len(file_data) > app.config['MAX_AUDIO_SIZE']:
                return jsonify({'success': False, 'error': '音频文件过大'})
            result = process_audio_file(file_data, filename)
            
        elif media_type == 'image':
            if len(file_data) > app.config['MAX_IMAGE_SIZE']:
                return jsonify({'success': False, 'error': '图片文件过大'})
            result = process_image_file(file_data, filename)
            
        elif media_type == 'video':
            if len(file_data) > app.config['MAX_VIDEO_SIZE']:
                return jsonify({'success': False, 'error': '视频文件过大'})
            result = process_video_file(file_data, filename)
            
        elif media_type == 'file':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'chat_files', filename)
            with open(file_path, 'wb') as f:
                f.write(file_data)
            result = {
                'success': True,
                'file_path': file_path,
                'file_name': filename,
                'file_size': len(file_data)
            }
        
        if result and result['success']:
            # 保存到聊天记录
            if session_id not in chat_sessions:
                chat_sessions[session_id] = []
            
            chat_sessions[session_id].append({
                'type': media_type,
                'content': result,
                'timestamp': datetime.now().isoformat(),
                'sender': 'user'
            })
        
        return jsonify(result or {'success': False, 'error': '不支持的媒体类型'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chat/tts', methods=['POST'])
def text_to_speech_api():
    """文本转语音API"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        session_id = data.get('session_id', 'default')
        
        if not text:
            return jsonify({'success': False, 'error': '文本不能为空'})
        
        result = text_to_speech(text, session_id)
        
        # 保存到聊天记录
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        chat_sessions[session_id].append({
            'type': 'text',
            'content': text,
            'response': result,
            'timestamp': datetime.now().isoformat(),
            'sender': 'ai'
        })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chat/history/<session_id>')
def get_chat_history(session_id):
    """获取聊天记录"""
    try:
        history = chat_sessions.get(session_id, [])
        return jsonify({
            'success': True,
            'history': history,
            'session_id': session_id
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chat/video/start', methods=['POST'])
def start_video_call():
    """开始视频通话"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        # 生成通话ID
        call_id = str(uuid.uuid4())
        
        return jsonify({
            'success': True,
            'call_id': call_id,
            'session_id': session_id,
            'status': 'connected',
            'webrtc_config': {
                'iceServers': [
                    {'urls': 'stun:stun.l.google.com:19302'}
                ]
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chat/audio/start', methods=['POST'])
def start_audio_call():
    """开始语音通话"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        # 生成通话ID
        call_id = str(uuid.uuid4())
        
        return jsonify({
            'success': True,
            'call_id': call_id,
            'session_id': session_id,
            'status': 'connected',
            'type': 'audio',
            'webrtc_config': {
                'iceServers': [
                    {'urls': 'stun:stun.l.google.com:19302'}
                ]
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})



@app.route('/profile')
def profile_page():
    """个人资料页面"""
    return render_template('profile.html')

@app.route('/preferences')
def preferences_page():
    """偏好设置页面"""
    return render_template('preferences.html')

@app.route('/knowledge_interface')
def knowledge_interface_page():
    """知识库专家界面"""
    return render_template('knowledge_interface.html')

# API路由 - 知识库管理
@app.route('/api/knowledge/list')
def api_knowledge_list():
    """获取知识库列表"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        category = request.args.get('category', '')
        search = request.args.get('search', '')
        
        items = list(knowledge_base.values())
        
        # 过滤
        if category:
            items = [item for item in items if item.get('category') == category]
        
        if search:
            items = [item for item in items if search.lower() in item.get('title', '').lower() or 
                     search.lower() in item.get('content', '').lower()]
        
        # 分页
        total = len(items)
        start = (page - 1) * per_page
        end = start + per_page
        items = items[start:end]
        
        return jsonify({
            'success': True,
            'items': items,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/knowledge/<knowledge_id>')
def api_knowledge_detail(knowledge_id):
    """获取知识详情"""
    try:
        if knowledge_id in knowledge_base:
            return jsonify({
                'success': True,
                'data': knowledge_base[knowledge_id]
            })
        else:
            return jsonify({'success': False, 'error': '知识条目不存在'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/knowledge/<knowledge_id>', methods=['DELETE'])
def api_delete_knowledge(knowledge_id):
    """删除知识条目"""
    try:
        if knowledge_id in knowledge_base:
            # 删除关联文件
            file_path = knowledge_base[knowledge_id].get('file_path')
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"删除文件失败: {e}")
            
            del knowledge_base[knowledge_id]
            save_knowledge_base()
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': '知识条目不存在'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/knowledge/<knowledge_id>', methods=['PUT'])
def api_update_knowledge(knowledge_id):
    """更新知识条目"""
    try:
        if knowledge_id not in knowledge_base:
            return jsonify({'success': False, 'error': '知识条目不存在'})
        
        data = request.json
        allowed_fields = ['title', 'category', 'content']
        
        for field in allowed_fields:
            if field in data:
                knowledge_base[knowledge_id][field] = data[field]
        
        knowledge_base[knowledge_id]['updated_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        save_knowledge_base()
        
        return jsonify({'success': True, 'data': knowledge_base[knowledge_id]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})



# 搜索API
@app.route('/api/search')
def api_search():
    """搜索知识库"""
    try:
        query = request.args.get('q', '')
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({'success': True, 'results': []})
        
        results = []
        for knowledge_id, item in knowledge_base.items():
            score = 0
            title = item.get('title', '').lower()
            content = item.get('content', '').lower()
            
            if query.lower() in title:
                score += 10
            if query.lower() in content:
                score += 1
            
            if score > 0:
                results.append({
                    'id': knowledge_id,
                    'title': item.get('title'),
                    'category': item.get('category'),
                    'file_type': item.get('file_type'),
                    'upload_time': item.get('upload_time'),
                    'score': score
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:limit]
        
        return jsonify({
            'success': True,
            'results': results,
            'query': query,
            'total': len(results)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/categories')
def api_categories():
    """获取所有分类"""
    try:
        categories = {}
        for item in knowledge_base.values():
            category = item.get('category', '未分类')
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        return jsonify({
            'success': True,
            'categories': [{'name': k, 'count': v} for k, v in categories.items()]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stats')
def api_stats():
    """获取统计数据"""
    try:
        total_files = len(knowledge_base)
        total_size = sum(item.get('file_size', 0) for item in knowledge_base.values())
        
        file_types = {}
        for item in knowledge_base.values():
            file_type = item.get('file_type', 'other')
            if file_type not in file_types:
                file_types[file_type] = 0
            file_types[file_type] += 1
        
        return jsonify({
            'success': True,
            'data': {
                'total_files': total_files,
                'total_size': total_size,
                'file_types': file_types,
                'categories': len(set(item.get('category', '未分类') for item in knowledge_base.values()))
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """提供上传的文件"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.errorhandler(404)
def page_not_found(e):
    """404错误处理 - 添加调试信息"""
    # 调试信息
    from flask import request
    print(f"404错误 - 请求的URL: {request.url}")
    print(f"请求方法: {request.method}")
    print(f"可用路由:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.rule} -> {rule.endpoint}")
    
    return render_template('knowledge_import.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """500错误处理"""
    logger.error(f"服务器错误: {str(e)}")
    return render_template('knowledge_import.html'), 500

# 添加工具方法
def is_float(self, value):
    """检查字符串是否为浮点数"""
    try:
        float(value)
        return True
    except ValueError:
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 A管理模型智能AI助手已启动")
    print("=" * 60)
    print("📁 知识库导入系统: http://localhost:8003")
    print("💬 AI对话界面: http://localhost:8003/chat")
    print("📊 数据分析: http://localhost:8003/analytics")
    print("⚙️  系统设置: http://localhost:8003/settings/general")
    print("❓ 帮助中心: http://localhost:8003/help")
    print("📱 API文档: http://localhost:8003/api/docs")
    print("=" * 60)
    print("🎉 系统已全面上线，欢迎使用！")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8003, debug=True)