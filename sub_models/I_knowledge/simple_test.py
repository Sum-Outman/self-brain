from knowledge_app import app

# 测试所有路由
print("=== 测试Flask应用内部路由 ===")

with app.test_client() as client:
    routes = ['/', '/knowledge', '/import', '/analytics', '/settings', '/chat']
    
    for route in routes:
        response = client.get(route)
        print(f"{route}: {response.status_code}")
        
        # 如果是404，打印更多信息
        if response.status_code == 404:
            print(f"  404详情: {response.data[:100]}")

print("=== 路由注册详情 ===")
from flask import url_for

with app.app_context():
    for rule in app.url_map.iter_rules():
        if 'knowledge' in str(rule):
            print(f"规则: {rule.rule} -> {rule.endpoint}")