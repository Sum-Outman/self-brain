#!/usr/bin/env python3
"""
知识库数据初始化脚本
"""

import os
import json
import sqlite3
from datetime import datetime

def init_database():
    """初始化数据库"""
    db_path = 'knowledge.db'
    
    # 如果数据库已存在，先备份
    if os.path.exists(db_path):
        backup_path = f'knowledge_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
        os.rename(db_path, backup_path)
        print(f"已备份旧数据库到: {backup_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建知识表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            content_length INTEGER NOT NULL,
            upload_time TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 创建索引
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON knowledge(category)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_type ON knowledge(file_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_upload_time ON knowledge(upload_time)')
    
    conn.commit()
    conn.close()
    print("数据库初始化完成")

def create_sample_data():
    """创建示例数据"""
    conn = sqlite3.connect('knowledge.db')
    cursor = conn.cursor()
    
    # 插入示例数据
    sample_data = [
        {
            'id': 'sample1',
            'filename': '人工智能简介.txt',
            'title': '人工智能基础概念',
            'content': '人工智能（AI）是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。AI技术包括机器学习、深度学习、自然语言处理等。',
            'category': '人工智能',
            'file_type': 'text',
            'file_size': 256,
            'content_length': 120,
            'upload_time': '2025-09-04 13:00:00'
        },
        {
            'id': 'sample2',
            'filename': '机器学习入门.pdf',
            'title': '机器学习基础教程',
            'content': '机器学习是人工智能的一个子领域，专注于开发能够从数据中学习并做出预测的算法。主要类型包括监督学习、无监督学习和强化学习。',
            'category': '机器学习',
            'file_type': 'document',
            'file_size': 1024,
            'content_length': 200,
            'upload_time': '2025-09-04 12:30:00'
        },
        {
            'id': 'sample3',
            'filename': '深度学习应用.jpg',
            'title': '深度学习在图像识别中的应用',
            'content': '深度学习使用多层神经网络来处理复杂的数据模式。在图像识别领域，卷积神经网络(CNN)被广泛用于人脸识别、物体检测等任务。',
            'category': '深度学习',
            'file_type': 'image',
            'file_size': 2048,
            'content_length': 180,
            'upload_time': '2025-09-04 12:00:00'
        }
    ]
    
    for item in sample_data:
        cursor.execute('''
            INSERT OR IGNORE INTO knowledge 
            (id, filename, title, content, category, file_type, file_size, content_length, upload_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item['id'], item['filename'], item['title'], item['content'],
            item['category'], item['file_type'], item['file_size'],
            item['content_length'], item['upload_time']
        ))
    
    conn.commit()
    conn.close()
    print("示例数据创建完成")

def create_directories():
    """创建必要的目录"""
    directories = [
        'uploads',
        'backups',
        'logs'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")

def main():
    """主函数"""
    print("开始初始化知识库系统...")
    
    # 创建目录
    create_directories()
    
    # 初始化数据库
    init_database()
    
    # 创建示例数据
    create_sample_data()
    
    print("初始化完成！")
    print("现在可以启动知识库导入系统了")

if __name__ == "__main__":
    main()