from knowledge_app import app
import requests
import time
import threading

def test_routes():
    """测试实际运行的服务器路由"""
    time.sleep(2)  # 等待服务器启动
    
    base_url = "http://localhost:8003"
    routes = ["/", "/knowledge", "/import", "/analytics", "/settings"]
    
    print("=== 测试实际运行的服务器路由 ===")
    for route in routes:
        try:
            response = requests.get(f"{base_url}{route}", timeout=5)
            print(f"{route}: {response.status_code} - {len(response.content)} bytes")
            if response.status_code == 404:
                print(f"  404响应内容预览: {response.text[:200]}...")
        except Exception as e:
            print(f"{route}: 错误 - {e}")

if __name__ == '__main__':
    # 在后台线程中测试
    test_thread = threading.Thread(target=test_routes)
    test_thread.start()
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=8003, debug=False)