import requests
import socket
import time

# 测试不同地址的连接
def test_connection(url):
    try:
        start_time = time.time()
        print(f"尝试连接到: {url}")
        response = requests.get(url, timeout=5)
        end_time = time.time()
        print(f"连接成功，状态码: {response.status_code}")
        print(f"响应时间: {end_time - start_time:.4f} 秒")
        
        # 尝试解析JSON响应
        try:
            json_data = response.json()
            print(f"JSON响应: {json_data}")
        except:
            print(f"文本响应: {response.text[:100]}...")
        
        return True
    except Exception as e:
        print(f"连接失败: {str(e)}")
        return False

# 测试socket连接
def test_socket(host, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"Socket连接到 {host}:{port} 成功")
            return True
        else:
            print(f"Socket连接到 {host}:{port} 失败，错误代码: {result}")
            return False
    except Exception as e:
        print(f"Socket测试异常: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== 健康检查测试 ===")
    
    # 测试socket连接
    hosts = ["localhost", "127.0.0.1", "0.0.0.0"]
    port = 5000
    
    for host in hosts:
        print(f"\n测试socket连接 {host}:{port}:")
        test_socket(host, port)
    
    # 测试HTTP连接
    urls = [
        "http://127.0.0.1:5000/health",
        "http://127.0.0.1:5000/api/system/status",
        "http://localhost:5000/health",
        "http://localhost:5000/api/system/status"
    ]
    
    for url in urls:
        print(f"\n测试HTTP连接 {url}:")
        test_connection(url)
    
    print("\n=== 测试完成 ===")