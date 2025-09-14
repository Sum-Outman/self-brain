#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试合并后的统一系统功能
Test unified system functionality after merging
"""

import requests
import json
import time
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_system_health():
    """测试系统健康检查"""
    try:
        response = requests.get('http://localhost:5015/api/health')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 健康检查通过: {data}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False

def test_system_status():
    """测试系统状态"""
    try:
        response = requests.get('http://localhost:5015/api/status')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 系统状态获取成功")
            return True
        else:
            print(f"❌ 系统状态获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 系统状态获取异常: {e}")
        return False

def test_collaboration_task():
    """测试协作任务创建"""
    try:
        task_data = {
            "description": "测试协作任务 - 分析文本情感并生成总结",
            "required_models": ["A_management", "B_language"],
            "priority": "high",
            "metadata": {"test": True, "user": "test_script"}
        }
        
        response = requests.post(
            'http://localhost:5015/api/collaboration/tasks',
            json=task_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 协作任务创建成功: {data}")
            return True
        else:
            print(f"❌ 协作任务创建失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 协作任务创建异常: {e}")
        return False

def test_collaboration_stats():
    """测试协作统计"""
    try:
        response = requests.get('http://localhost:5015/api/collaboration/stats')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 协作统计获取成功")
            return True
        else:
            print(f"❌ 协作统计获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 协作统计获取异常: {e}")
        return False

def test_optimization_stats():
    """测试优化统计"""
    try:
        response = requests.get('http://localhost:5015/api/optimization/stats')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 优化统计获取成功")
            return True
        else:
            print(f"❌ 优化统计获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 优化统计获取异常: {e}")
        return False

def test_message_processing():
    """测试消息处理"""
    try:
        message_data = {
            "message": "这是一个测试消息，请分析我的情感状态",
            "task_type": "emotional_analysis",
            "emotional_context": {"mood": "neutral"}
        }
        
        response = requests.post(
            'http://localhost:5015/process_message',
            json=message_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 消息处理成功")
            return True
        else:
            print(f"❌ 消息处理失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 消息处理异常: {e}")
        return False

def test_models_list():
    """测试模型列表"""
    try:
        response = requests.get('http://localhost:5015/api/models')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 模型列表获取成功: {len(data.get('models', []))} 个模型")
            return True
        else:
            print(f"❌ 模型列表获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 模型列表获取异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试合并后的统一系统...")
    print("=" * 50)
    
    tests = [
        ("健康检查", test_system_health),
        ("系统状态", test_system_status),
        ("模型列表", test_models_list),
        ("消息处理", test_message_processing),
        ("协作任务", test_collaboration_task),
        ("协作统计", test_collaboration_stats),
        ("优化统计", test_optimization_stats)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 测试: {test_name}")
        if test_func():
            passed += 1
        time.sleep(0.5)  # 避免过快请求
    
    print("\n" + "=" * 50)
    print(f"🎯 测试结果: {passed}/{total} 项通过")
    
    if passed == total:
        print("🎉 所有测试通过！合并后的系统运行正常")
    else:
        print("⚠️  部分测试失败，请检查系统状态")

if __name__ == "__main__":
    main()