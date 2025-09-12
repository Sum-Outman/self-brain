#!/usr/bin/env python3
"""
Script to replace Chinese content with English in app.py
"""

import re
import os

def replace_chinese_in_file(file_path):
    """Replace Chinese content with English equivalents"""
    
    # Chinese to English mapping for app.py
    replacements = {
        # Comments and labels
        r'# 保存文件': '# Save file',
        r'# 生成文件信息': '# Generate file information',
        r'# 调用相应的AI模型': '# Call appropriate AI model',
        r'# 发送输入指示': '# Send input indication',
        r'# 处理文件附件': '# Process file attachments',
        r'# 这里可以处理实际的文件内容分析': '# Here you can process actual file content analysis',
        r'# 模拟AI处理时间（实际应用中调用模型API）': '# Simulate AI processing time (call model API in actual application)',
        r'# 增加处理时间以模拟复杂分析': '# Increase processing time to simulate complex analysis',
        r'# 生成智能响应': '# Generate intelligent response',
        r'# 添加文件处理结果的响应': '# Add file processing results response',
        r'已处理 (\d+) 个文件：': r'Processed \1 files:',
        
        # API responses and messages
        r'文件上传成功 \| File uploaded successfully': 'File uploaded successfully',
        r'没有选择ZIP文件 \| No ZIP file selected': 'No ZIP file selected',
        r'ZIP导入完成: (\d+)/(\d+) 个文件成功 \| ZIP import completed': r'ZIP import completed: \1/\2 files successful',
        r'无效的JSON配置 \| Invalid JSON configuration': 'Invalid JSON configuration',
        r'配置文件中没有文件列表 \| No file list in configuration': 'No file list in configuration',
        r'文件不存在: (.+)': r'File does not exist: \1',
        r'处理文件失败: (.+)': r'Failed to process file: \1',
        r'JSON配置导入完成: (\d+)/(\d+) 个文件成功 \| JSON import completed': r'JSON configuration import completed: \1/\2 files successful',
        r'处理文件 (.+) 失败: (.+)': r'Failed to process file \1: \2',
        r'删除知识: (.+) \| Deleting knowledge: (.+)': r'Deleting knowledge: \1',
        r'成功删除文件: (.+) \| Successfully deleted file: (.+)': r'Successfully deleted file: \1',
        r'知识删除成功 \| Knowledge deleted successfully': 'Knowledge deleted successfully',
        r'删除文件失败: (.+) \| Failed to delete file: (.+)': r'Failed to delete file: \1',
        r'删除失败: (.+)': r'Deletion failed: \1',
        r'文件不存在 \| File not found': 'File not found',
        r'未找到指定的知识文件 \| Knowledge file not found': 'Knowledge file not found',
        r'没有选择要删除的项目': 'No items selected for deletion',
        r'成功删除 (\d+) 个文件': r'Successfully deleted \1 files',
        r'，(\d+) 个文件删除失败': r', \1 files failed to delete',
        r'未找到可删除的文件': 'No files found to delete',
        r'没有选择要导出的项目': 'No items selected for export',
        r'未找到选中的文件': 'Selected files not found',
        
        # Knowledge base content
        r'机器学习基础 \| Machine Learning Basics': 'Machine Learning Basics',
        r'技术 \| Technology': 'Technology',
        r'机器学习是人工智能的一个分支': 'Machine learning is a branch of artificial intelligence',
        r'机器学习': 'machine learning',
        r'人工智能': 'artificial intelligence',
        r'数据科学': 'data science',
        
        # SocketIO events and logging
        r'# SocketIO 事件处理 \| SocketIO event handling': '# SocketIO event handling',
        r'"""客户端连接事件 \| Client connect event"""': '"""Client connect event"""',
        r'客户端连接: (.+) \| Client connected: (.+)': r'Client connected: \1',
        r'"""客户端断开连接事件 \| Client disconnect event"""': '"""Client disconnect event"""',
        r'客户端断开连接: (.+) \| Client disconnected: (.+)': r'Client disconnected: \1',
        r'"""处理仪表盘更新请求 \| Handle dashboard update request"""': '"""Handle dashboard update request"""',
        r'仪表盘更新请求失败: (.+) \| Dashboard update request failed: (.+)': r'Dashboard update request failed: \1',
        r'"""处理训练状态请求 \| Handle training status request"""': '"""Handle training status request"""',
        r'获取训练状态失败: (.+) \| Failed to get training status: (.+)': r'Failed to get training status: \1',
        r'"""处理模型状态请求 \| Handle model status request"""': '"""Handle model status request"""',
        r'获取模型状态失败: (.+) \| Failed to get model status: (.+)': r'Failed to get model status: \1',
        r'"""处理所有模型状态请求 \| Handle all models status request"""': '"""Handle all models status request"""',
        r'获取所有模型状态失败: (.+) \| Failed to get all models status: (.+)': r'Failed to get all models status: \1',
        r'# 后台任务：定期广播系统状态 \| Background task: periodically broadcast system status': '# Background task: periodically broadcast system status',
        r'"""后台广播系统状态 \| Background broadcast system status"""': '"""Background broadcast system status"""',
        r'# 通过增强版监控系统获取最新数据 \| Get latest data through enhanced monitoring system': '# Get latest data through enhanced monitoring system',
        r'# 使用SocketIO事件触发监控系统发送数据': '# Use SocketIO events to trigger monitoring system to send data',
        r'# 每3秒更新一次 \| Update every 3 seconds': '# Update every 3 seconds',
        r'# 使用安全的日志记录方式，避免格式化字符问题': '# Use safe logging method to avoid format character issues',
        r'# 使用字符串拼接而不是格式化字符串': '# Use string concatenation instead of format strings',
        r'后台广播失败: (.+) \| Background broadcast failed: (.+)': r'Background broadcast failed: \1',
        r'# 启动后台广播线程 \| Start background broadcast thread': '# Start background broadcast thread',
        r'# 错误处理 \| Error handling': '# Error handling',
        r'"""404错误处理 \| 404 error handling"""': '"""404 error handling"""',
        r'# 使用硬编码的语言资源以避免任何加载错误 \| Use hardcoded language resources to avoid any loading errors': '# Use hardcoded language resources to avoid any loading errors',
        r'error_message="页面未找到 \| Page not found"': 'error_message="Page not found"',
        r'"""500错误处理 \| 500 error handling"""': '"""500 error handling"""',
        r'error_message="服务器内部错误 \| Internal server error"': 'error_message="Internal server error"',
        r'# 知识编辑相关API \| Knowledge edit related APIs': '# Knowledge edit related APIs',
        r'# 支持JSON格式数据': '# Support JSON format data',
        r'# 兼容表单数据格式': '# Compatible with form data format',
        r'# 生成知识ID': '# Generate knowledge ID',
        r'# 创建目录': '# Create directory',
        r'# 保存内容': '# Save content',
        r'# 保存元数据': '# Save metadata',
        r'# 处理上传的文件（仅表单提交时处理）': '# Process uploaded files (only for form submission)',
        r'# 更新文件计数': '# Update file count',
        r'# 处理表单数据': '# Process form data',
        r'# 更新内容': '# Update content',
        r'# 更新元数据': '# Update metadata',
        r'# 处理新上传的文件': '# Process newly uploaded files',
        r'"""查看知识页面 \| View knowledge page"""': '"""View knowledge page"""',
        r'"""编辑知识页面 \| Edit knowledge page"""': '"""Edit knowledge page"""',
        r'"""创建新知识页面 \| Create new knowledge page"""': '"""Create new knowledge page"""',
        r'"""优化知识数据库 \| Optimize knowledge database"""': '"""Optimize knowledge database"""',
        r'# 检查知识数据库文件': '# Check knowledge database file',
        r'# 调试信息': '# Debug information',
        r'logger.info("开始优化知识数据库 \| Starting knowledge database optimization")': 'logger.info("Starting knowledge database optimization")',
        r'# 如果数据库不存在，创建它': '# If database does not exist, create it',
        r'# 创建数据库和表结构': '# Create database and table structure',
        r'# 创建知识表': '# Create knowledge table',
        r'# 创建索引': '# Create index',
        r'logger.info("知识数据库已创建 \| Knowledge database created")': 'logger.info("Knowledge database created")',
        r'# 创建备份': '# Create backup',
        r'logger.info(f"数据库备份已创建: {backup_file}")': 'logger.info(f"Database backup created: {backup_file}")',
        r'# 连接数据库并执行优化': '# Connect to database and perform optimization',
        r'# 执行VACUUM优化': '# Execute VACUUM optimization',
        r'# 重新索引': '# Reindex',
        r'logger.info("数据库优化完成 \| Database optimization completed")': 'logger.info("Database optimization completed")',
        r'# 确保知识库存储目录存在': '# Ensure knowledge base storage directory exists',
        r'logger.info("存储目录已创建 \| Storage directory created")': 'logger.info("Storage directory created")',
        r'# 清理知识库存储中的空文件夹': '# Clean empty folders in knowledge base storage',
        r'# 检查文件夹是否为空': '# Check if folder is empty',
        r'logger.warning(f"清理文件夹失败: {cleanup_error}")': 'logger.warning(f"Failed to clean folder: {cleanup_error}")',
        r'logger.info(f"已清理 {cleaned_folders} 个空文件夹")': 'logger.info(f"Cleaned {cleaned_folders} empty folders")',
        r'logger.warning(f"清理存储目录时出错: {e}")': 'logger.warning(f"Error cleaning storage directory: {e}")',
        r'logger.info(f"知识数据库优化完成: {final_message}")': 'logger.info(f"Knowledge database optimization completed: {final_message}")',
        r'error_msg = f"优化知识数据库失败: {str(e)}"': 'error_msg = f"Knowledge database optimization failed: {str(e)}"',
        r'# 启动应用 \| Start application': '# Start application',
        r'# 启动A管理模型API': '# Start A management model API',
        r'logger.info("启动 Self Brain AGI 系统 Web 接口 \| Starting Self Brain AGI System Web Interface")': 'logger.info("Starting Self Brain AGI System Web Interface")',
        r'logger.info("访问 http://localhost:5000 查看主页面 \| Visit http://localhost:5000 for main page")': 'logger.info("Visit http://localhost:5000 for main page")',
        r'# 运行Flask应用 \| Run Flask application': '# Run Flask application',
        
        # Voice call messages
        r'通话已启动': 'Call started',
        r'您好！我是Self Brain AGI语音助手，我们可以开始语音对话了。请对着麦克风说话，我会实时回答您的问题。': 'Hello! I am Self Brain AGI voice assistant. We can start voice conversation now. Please speak into the microphone and I will answer your questions in real time.',
        r'视频通话已连接！我是您的AI助手，可以通过视频为您提供更直观的帮助。您可以看到我的实时回应。': 'Video call connected! I am your AI assistant and can provide more intuitive help through video. You can see my real-time responses.',
        r'通话已连接': 'Call connected',
        r'感谢您的使用！如有需要，随时可以再次发起通话。': 'Thank you for using! Feel free to start a call again anytime you need.',
        r'通话已结束': 'Call ended',
        
        # Model labels and categories
        r'默认模型': 'default model',
        r'计算存储空间使用情况': 'Calculate storage usage',
        r'获取目录结构': 'Get directory structure',
        r'计算各分类文件数量': 'Calculate file count by category',
        r'遍历所有文件': 'Iterate through all files',
        r'根据文件路径确定分类': 'Determine category based on file path',
        r'获取查询参数': 'Get query parameters',
        r'创建知识条目': 'Create knowledge entry',
        r'应用过滤条件': 'Apply filtering criteria',
        r'计算分页': 'Calculate pagination',
        r'获取当前页数据': 'Get current page data',
        r'计算统计信息': 'Calculate statistics',
        r'确保存储路径存在': 'Ensure storage path exists',
        r'生成唯一文件名': 'Generate unique filename',
        r'保存文件': 'Save file',
        r'计算文件信息': 'Calculate file information',
        r'确定分类': 'Determine category',
        r'创建分类目录': 'Create category directory',
        r'生成目标路径': 'Generate target path',
        r'复制文件': 'Copy file',
        r'统计文件类型': 'Count file types',
        r'排除的系统文件列表': 'Excluded system file list',
        r'跳过系统文件': 'Skip system files',
        r'根据文件扩展名分类统计': 'Count by file extension category',
        r'建立ID到文件路径的映射': 'Build mapping from ID to file path',
        r'支持更多文件类型': 'Support more file types',
        r'清理临时文件': 'Clean temporary files',
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Perform replacements
    for chinese_pattern, english_replacement in replacements.items():
        content = re.sub(chinese_pattern, english_replacement, content, flags=re.MULTILINE)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Successfully replaced Chinese content in {file_path}")

if __name__ == "__main__":
    app_py_path = r"d:\shiyan\web_interface\app.py"
    replace_chinese_in_file(app_py_path)
    
    # Also process other files with Chinese content
    files_to_process = [
        r"d:\shiyan\web_interface\app_fixed.py",
        r"d:\shiyan\web_interface\static\css\style.css",
    ]
    
    for file_path in files_to_process:
        if os.path.exists(file_path):
            replace_chinese_in_file(file_path)
            print(f"Processed: {file_path}")