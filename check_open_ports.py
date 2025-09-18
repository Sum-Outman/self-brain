import socket
import time

# 检查特定端口是否开放
def check_port(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        result = sock.connect_ex((host, port))
        return result == 0  # 0表示端口开放
    finally:
        sock.close()

# 检查系统上所有相关端口
def check_system_ports():
    print("检查Self Brain AGI系统相关端口...")
    
    # 检查主要服务端口
    ports_to_check = [
        5000,  # Main Web Interface
        5001,  # A Management Model
        5002,  # B Language Model
        5003,  # C Audio Model
        5004,  # D Image Model
        5005,  # E Video Model
        5006,  # F Spatial Model
        5007,  # G Sensor Model
        5008,  # H Computer Control
        5009,  # I Knowledge Model
        5010,  # J Motion Model
        5011,  # K Programming Model
        5015,  # Manager Model API
        5016   # Working Enhanced Chat
    ]
    
    open_ports = []
    closed_ports = []
    
    for port in ports_to_check:
        if check_port('localhost', port):
            open_ports.append(port)
        else:
            closed_ports.append(port)
    
    print(f"开放的端口: {open_ports}")
    print(f"关闭的端口: {closed_ports}")
    
    # 特别检查5000端口的详细状态
    if 5000 in open_ports:
        print("\n5000端口开放，但可能是由其他进程占用。让我尝试使用系统命令查看端口占用情况。")
        
        try:
            import subprocess
            # 使用netstat命令查看端口占用情况
            if hasattr(subprocess, 'run'):
                # Python 3.5+
                result = subprocess.run(['netstat', '-ano', '|', 'findstr', ':5000'], 
                                       capture_output=True, text=True, shell=True)
                print(f"netstat输出: {result.stdout}")
            else:
                print("无法执行netstat命令：Python版本过低")
        except Exception as e:
            print(f"执行netstat命令时出错: {e}")
    else:
        print("\n5000端口未开放，Web界面服务可能未正确启动。")

if __name__ == "__main__":
    check_system_ports()