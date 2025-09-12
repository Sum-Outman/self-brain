from flask import Flask, render_template

# 创建最小化的Flask应用
app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return "Home page"

@app.route('/knowledge')
def knowledge_page():
    """知识库管理页面"""
    return render_template('knowledge_manage.html')

@app.route('/test')
def test():
    return "Test page"

if __name__ == '__main__':
    print("=== 最小化测试应用 ===")
    print("路由测试:")
    with app.test_client() as client:
        for route in ['/', '/knowledge', '/test']:
            response = client.get(route)
            print(f"{route}: {response.status_code}")
    
    print("\n启动服务器...")
    app.run(host='0.0.0.0', port=8004, debug=True)