#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终验证：确认training_control.get_system_health()的实际返回数据
"""
import sys
import os
import json
import requests
import time

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath('.')))

def test_training_controller_direct():
    """直接测试训练控制器的get_system_health方法"""
    print("=== 训练控制器直接测试 ===")
    
    from training_manager.advanced_train_control import get_training_controller
    
    # 获取训练控制器实例
    controller = get_training_controller()
    
    # 直接调用get_system_health
    health_data = controller.get_system_health()
    
    print("完整健康数据结构:")
    print(json.dumps(health_data, indent=2, ensure_ascii=False))
    
    # 检查system字段
    system = health_data.get('system', {})
    print(f"\nSystem字段:")
    print(f"  所有键: {list(system.keys())}")
    print(f"  包含gpu_usage_percent: {'gpu_usage_percent' in system}")
    print(f"  包含gpu_model: {'gpu_model' in system}")
    
    if 'gpu_usage_percent' in system:
        print(f"  gpu_usage_percent值: {system['gpu_usage_percent']}")
    if 'gpu_model' in system:
        print(f"  gpu_model值: {system['gpu_model']}")

def test_api_vs_direct_comparison():
    """对比API响应与直接调用"""
    print("\n=== API vs 直接调用对比 ===")
    
    # 1. 直接调用
    from training_manager.advanced_train_control import get_training_controller
    controller = get_training_controller()
    direct_health = controller.get_system_health()
    direct_system = direct_health.get('system', {})
    
    # 2. API调用
    try:
        response = requests.get('http://localhost:5000/api/system/resources', timeout=5)
        if response.status_code == 200:
            api_data = response.json()
            api_system = api_data.get('resources', {}).get('system', {})
            
            print("直接调用system字段:")
            print(f"  {list(direct_system.keys())}")
            print(f"  gpu_usage_percent: {direct_system.get('gpu_usage_percent')}")
            print(f"  gpu_model: {direct_system.get('gpu_model')}")
            
            print("\nAPI调用system字段:")
            print(f"  {list(api_system.keys())}")
            print(f"  gpu_usage_percent: {api_system.get('gpu_usage_percent')}")
            print(f"  gpu_model: {api_system.get('gpu_model')}")
            
            # 找出差异
            missing_in_api = set(direct_system.keys()) - set(api_system.keys())
            extra_in_api = set(api_system.keys()) - set(direct_system.keys())
            
            print(f"\n差异分析:")
            print(f"  API缺失字段: {missing_in_api}")
            print(f"  API额外字段: {extra_in_api}")
            
        else:
            print(f"API响应状态码: {response.status_code}")
    except Exception as e:
        print(f"API调用失败: {e}")

def test_method_source_inspection():
    """检查get_system_health方法的源代码"""
    print("\n=== 方法源码检查 ===")
    
    from training_manager.advanced_train_control import get_training_controller
    import inspect
    
    controller = get_training_controller()
    method = controller.get_system_health
    
    # 获取源码
    try:
        source = inspect.getsource(method)
        print("方法源码（关键部分）:")
        
        # 查找GPU相关代码
        lines = source.split('\n')
        gpu_lines = []
        for i, line in enumerate(lines):
            if 'gpu' in line.lower() or 'GPUtil' in line:
                gpu_lines.append(f"{i+1}: {line.strip()}")
        
        for line in gpu_lines[:10]:  # 显示前10行
            print(f"  {line}")
            
    except Exception as e:
        print(f"无法获取源码: {e}")

def test_real_time_api():
    """实时测试API响应"""
    print("\n=== 实时API测试 ===")
    
    # 使用Flask测试客户端
    from web_interface.app import app
    
    with app.test_client() as client:
        response = client.get('/api/system/resources')
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.get_json()
            resources = data.get('resources', {})
            system = resources.get('system', {})
            
            print("实时API system字段:")
            for key, value in system.items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    test_training_controller_direct()
    test_api_vs_direct_comparison()
    test_method_source_inspection()
    test_real_time_api()