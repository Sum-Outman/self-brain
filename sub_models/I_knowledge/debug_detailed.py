from knowledge_app import app
import os

print("=== Flask 路由调试信息 ===")

# 检查Flask应用配置
print(f"模板文件夹路径: {app.template_folder}")
print(f"静态文件夹路径: {app.static_folder}")
print(f"根路径: {app.root_path}")

# 检查模板文件是否存在
template_path = os.path.join(app.template_folder, 'knowledge_manage.html')
print(f"knowledge_manage.html 路径: {template_path}")
print(f"模板文件存在: {os.path.exists(template_path)}")

# 打印所有注册的路由
print("\n=== 注册的路由 ===")
for rule in app.url_map.iter_rules():
    print(f"路径: {rule.rule} -> 端点: {rule.endpoint} -> 方法: {list(rule.methods)}")

# 测试路由匹配
print("\n=== 路由匹配测试 ===")
with app.test_request_context('/knowledge'):
    print(f"/knowledge 匹配结果: {app.url_map.bind('localhost').match('/knowledge')}")

# 测试模板渲染
print("\n=== 模板渲染测试 ===")
try:
    with app.app_context():
        from flask import render_template
        result = render_template('knowledge_manage.html')
        print("knowledge_manage.html 渲染成功")
        print(f"模板内容长度: {len(result)} 字符")
except Exception as e:
    print(f"模板渲染失败: {e}")

print("\n=== 调试完成 ===")