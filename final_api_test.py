#!/usr/bin/env python3
import requests
import json

try:
    response = requests.get('http://localhost:5000/api/system/resources', timeout=5)
    data = response.json()
    
    print('=== 完整API响应 ===')
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
    print('\n=== system字段详细内容 ===')
    system = data.get('resources', {}).get('system', {})
    for key, value in system.items():
        print(f'{key}: {value}')
        
    print('\n=== GPU字段检查结果 ===')
    print(f'gpu_usage_percent存在: {"gpu_usage_percent" in system}')
    print(f'gpu_model存在: {"gpu_model" in system}')
    
    if "gpu_usage_percent" in system:
        print(f'gpu_usage_percent值: {system["gpu_usage_percent"]}')
    if "gpu_model" in system:
        print(f'gpu_model值: {system["gpu_model"]}')
        
except Exception as e:
    print(f'错误: {e}')