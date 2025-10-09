#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from knowledge_app import app

def debug_app():
    print("=== Flask 应用调试信息 ===")
    
    # 检查所有路由
    print("\n1. 注册的路由:")
    with app.app_context():
        for rule in app.url_map.iter_rules():
            print(f"  {rule.rule} -> {rule.endpoint} ({','.join(rule.methods)})")
    
    # 检查端点映射
    print("\n2. 端点映射:")
    for endpoint, func in app.view_functions.items():
        print(f"  {endpoint} -> {func.__name__}")
    
    # 测试路由
    print("\n3. 路由测试:")
    with app.test_client() as client:
        test_routes = [
            '/',
            '/knowledge',
            '/import',
            '/analytics',
            '/settings',
            '/api/knowledge_list'
        ]
        
        for route in test_routes:
            response = client.get(route)
            print(f"  {route}: {response.status_code}")

if __name__ == '__main__':
    debug_app()