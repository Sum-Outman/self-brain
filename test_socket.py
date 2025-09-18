import socket
import time

# 创建一个socket连接测试函数
def test_socket_connection(host, port):
    try:
        # 创建socket对象
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置超时时间
        sock.settimeout(2)
        
        # 尝试连接
        result = sock.connect_ex((host, port))
        
        if result == 0:
            print(f"成功连接到 {host}:{port}")
            return True
        else:
            print(f"无法连接到 {host}:{port}，错误代码: {result}")
            return False
    except Exception as e:
        print(f"连接 {host}:{port} 时发生异常: {str(e)}")
        return False
    finally:
        sock.close()

# 创建一个简单的socket服务器测试端口绑定
def start_simple_server(host, port):
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置地址重用
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # 尝试绑定端口
        server_socket.bind((host, port))
        print(f"成功绑定到 {host}:{port}")
        
        # 开始监听
        server_socket.listen(1)
        print(f"服务器正在 {host}:{port} 监听...")
        
        # 启动一个简单的连接接受循环
        def accept_connection():
            try:
                # 设置超时以便我们可以定期检查是否需要停止
                server_socket.settimeout(1)
                conn, addr = server_socket.accept()
                print(f"接受到来自 {addr} 的连接")
                conn.close()
            except socket.timeout:
                # 超时是正常的，只是为了定期检查是否需要停止
                pass
            except Exception as e:
                print(f"接受连接时发生异常: {str(e)}")
        
        # 运行一小段时间
        start_time = time.time()
        while time.time() - start_time < 5:
            accept_connection()
        
        server_socket.close()
        print(f"服务器已停止")
        return True
    except Exception as e:
        print(f"启动服务器时发生异常: {str(e)}")
        server_socket.close()
        return False

if __name__ == "__main__":
    print("=== Socket连接测试 ===")
    
    # 测试不同主机地址的连接
    hosts = ["localhost", "127.0.0.1"]
    port = 5000
    
    for host in hosts:
        print(f"\n测试连接到 {host}:{port}:")
        test_socket_connection(host, port)
    
    # 测试启动服务器
    print("\n=== 测试启动简单服务器 ===")
    
    for host in hosts:
        print(f"\n尝试在 {host}:{port} 启动服务器:")
        if start_simple_server(host, port):
            # 如果成功启动，尝试连接
            print(f"\n测试连接到刚刚启动的服务器 {host}:{port}:")
            test_socket_connection(host, port)
            break

    print("\n=== 测试完成 ===")