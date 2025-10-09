#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库导入系统主程序
集成了文件上传、知识管理和AI对话功能
"""

from knowledge_import_routes import app

if __name__ == '__main__':
    print("🚀 知识库导入系统启动成功！")
    print("📊 访问地址：http://localhost:8003")
    print("📁 导入页面：http://localhost:8003")
    print("🤖 AI对话：http://localhost:8003/chat")
    print("📈 API文档：http://localhost:8003/api/docs")
    print("📋 知识列表：http://localhost:8003/api/knowledge_list")
    print("📊 统计信息：http://localhost:8003/api/statistics")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=8003, debug=True, use_reloader=False)