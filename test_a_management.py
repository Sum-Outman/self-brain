#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试A_management模型API服务
"""

import requests
import json
import time

def test_api():
    """测试API端点"""
    base_url = "http://localhost:5001"
    
    print("🚀 开始测试A_management模型API服务...")
    
    # 1. 测试健康检查
    print("\n1. 测试健康检查...")
    try:
        response = requests.get(f"{base_url}/api/health")
        print(f"✅ 健康检查: {response.json()}")
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return
    
    # 2. 测试系统状态
    print("\n2. 测试系统状态...")
    try:
        response = requests.get(f"{base_url}/api/status")
        data = response.json()
        if data['status'] == 'success':
            print(f"✅ 系统状态: {data['data']['status']}")
        else:
            print(f"❌ 系统状态错误: {data}")
    except Exception as e:
        print(f"❌ 系统状态测试失败: {e}")
    
    # 3. 测试模型列表
    print("\n3. 测试模型列表...")
    try:
        response = requests.get(f"{base_url}/api/models")
        data = response.json()
        if data['status'] == 'success':
            print(f"✅ 可用模型 ({data['count']}): {data['models']}")
        else:
            print(f"❌ 模型列表错误: {data}")
    except Exception as e:
        print(f"❌ 模型列表测试失败: {e}")
    
    # 4. 测试process_message端点
    print("\n4. 测试process_message端点...")
    test_messages = [
        {"message": "Hello, how are you?", "task_type": "general"},
        {"message": "What is machine learning?", "task_type": "knowledge"},
        {"message": "Can you help me with Python programming?", "task_type": "programming"},
    ]
    
    for msg in test_messages:
        try:
            response = requests.post(f"{base_url}/process_message", json=msg)
            data = response.json()
            
            if data['status'] == 'success':
                print(f"✅ 消息处理成功: {data['response'][:100]}...")
                print(f"   任务ID: {data['task_id']}")
                print(f"   使用模型: {data['models_used']}")
                print(f"   处理时间: {data['processing_time']:.2f}s")
            else:
                print(f"❌ 消息处理失败: {data}")
        except Exception as e:
            print(f"❌ 消息处理测试失败: {e}")
    
    # 5. 测试情感分析
    print("\n5. 测试情感分析...")
    try:
        response = requests.post(f"{base_url}/api/emotion/analyze", json={
            "text": "I love this amazing technology! It's wonderful and makes me happy."
        })
        data = response.json()
        if data['status'] == 'success':
            print(f"✅ 情感分析: {data['emotion']} (强度: {data['intensity']})")
        else:
            print(f"❌ 情感分析错误: {data}")
    except Exception as e:
        print(f"❌ 情感分析测试失败: {e}")
    
    # 6. 测试知识库查询
    print("\n6. 测试知识库查询...")
    try:
        response = requests.post(f"{base_url}/api/knowledge/query", json={
            "query": "What is artificial intelligence?",
            "domain": "technology"
        })
        data = response.json()
        if data['status'] == 'success':
            print(f"✅ 知识库响应: {data['response'][:100]}...")
        else:
            print(f"❌ 知识库查询错误: {data}")
    except Exception as e:
        print(f"❌ 知识库查询测试失败: {e}")
    
    # 7. 测试系统统计
    print("\n7. 测试系统统计...")
    try:
        response = requests.get(f"{base_url}/api/system/stats")
        data = response.json()
        if data['status'] == 'success':
            stats = data['stats']
            print(f"✅ 系统统计:")
            print(f"   总任务数: {stats['total_tasks_processed']}")
            print(f"   成功任务: {stats['successful_tasks']}")
            print(f"   失败任务: {stats['failed_tasks']}")
            print(f"   平均处理时间: {stats['average_processing_time']:.2f}s")
            print(f"   活跃模型: {stats['active_models']}")
        else:
            print(f"❌ 系统统计错误: {data}")
    except Exception as e:
        print(f"❌ 系统统计测试失败: {e}")
    
    print("\n🎉 测试完成！")

if __name__ == "__main__":
    # 等待服务启动
    time.sleep(2)
    test_api()