#!/usr/bin/env python3
"""
简化版A_management模型API测试脚本
"""

import requests
import json

def test_api():
    """测试所有API端点"""
    base_url = "http://localhost:5001"
    
    print("=== 测试A_management模型API ===")
    
    # 1. 测试健康检查
    print("\n1. 测试健康检查...")
    try:
        response = requests.get(f"{base_url}/api/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 2. 测试模型列表
    print("\n2. 测试模型列表...")
    try:
        response = requests.get(f"{base_url}/api/models")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 3. 测试process_message
    print("\n3. 测试process_message...")
    try:
        payload = {
            "message": "Hello, how are you?",
            "task_type": "general"
        }
        response = requests.post(f"{base_url}/process_message", json=payload)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print(f"响应: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"错误响应: {response.text}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 4. 测试情感分析
    print("\n4. 测试情感分析...")
    try:
        payload = {
            "text": "I am very happy today!"
        }
        response = requests.post(f"{base_url}/api/emotion/analyze", json=payload)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print(f"响应: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"错误响应: {response.text}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 5. 测试系统统计
    print("\n5. 测试系统统计...")
    try:
        response = requests.get(f"{base_url}/api/system/stats")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print(f"响应: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"错误响应: {response.text}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    test_api()