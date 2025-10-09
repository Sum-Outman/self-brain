#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self Brain AGI - A Management Model Startup Script
This script starts the Management Model API service.
"""

import os
import sys
import subprocess
import time
import requests
import json

# 设置项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 配置服务参数
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_PORT = 5000
SERVICE_URL = f"http://localhost:{SERVICE_PORT}"

# 检查端口是否被占用
def check_port(port):
    """检查端口是否被占用"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("localhost", port))
    sock.close()
    return result == 0

# 杀死占用端口的进程
def kill_process_on_port(port):
    """杀死占用指定端口的进程"""
    try:
        # Windows系统
        if os.name == 'nt':
            cmd = f'netstat -ano | findstr :{port}'
            result = subprocess.check_output(cmd, shell=True).decode()
            for line in result.split('\n'):
                if ':{}'.format(port) in line and 'LISTENING' in line:
                    pid = line.strip().split()[-1]
                    print(f"Killing process {pid} that is using port {port}")
                    subprocess.call(f'taskkill /F /PID {pid}', shell=True)
                    time.sleep(2)
                    return True
        return False
    except Exception as e:
        print(f"Error killing process on port {port}: {str(e)}")
        return False

# 启动管理模型服务
def start_management_service():
    """启动管理模型服务"""
    try:
        # 检查端口是否被占用
        if check_port(SERVICE_PORT):
            print(f"Port {SERVICE_PORT} is already in use. Trying to kill the process...")
            if not kill_process_on_port(SERVICE_PORT):
                print(f"Failed to free port {SERVICE_PORT}. Please free it manually.")
                return None
        
        # 启动服务
        print(f"Starting Management Model API service on port {SERVICE_PORT}...")
        
        # 使用不同的方式启动服务，根据操作系统
        if os.name == 'nt':  # Windows
            cmd = f'python {os.path.join(SERVICE_DIR, "app.py")} serve --port {SERVICE_PORT}'
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:  # Unix-like systems
            cmd = ['python', os.path.join(SERVICE_DIR, 'app.py'), 'serve', '--port', str(SERVICE_PORT)]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待服务启动
        time.sleep(5)
        
        # 检查服务是否正常运行
        try:
            response = requests.get(f"{SERVICE_URL}/health", timeout=5)
            if response.status_code == 200:
                print(f"Management Model service started successfully on {SERVICE_URL}")
                return proc
            else:
                print(f"Service started but returned status code {response.status_code}")
                return None
        except requests.exceptions.ConnectionError:
            print("Failed to connect to the Management Model service. It might not have started properly.")
            return None
    except Exception as e:
        print(f"Error starting Management Model service: {str(e)}")
        return None

# 测试服务
def test_management_service():
    """测试管理模型服务的基本功能"""
    try:
        print("Testing Management Model service...")
        
        # 测试健康检查端点
        print("Testing /health endpoint...")
        response = requests.get(f"{SERVICE_URL}/health", timeout=5)
        print(f"Health check response: {response.status_code} {response.text}")
        
        # 测试预测端点
        print("Testing /predict endpoint...")
        test_data = {"input": [0.1] * 100}
        response = requests.post(f"{SERVICE_URL}/predict", json=test_data, timeout=10)
        print(f"Prediction response: {response.status_code} {response.text}")
        
        return True
    except Exception as e:
        print(f"Error testing Management Model service: {str(e)}")
        return False

if __name__ == "__main__":
    # 启动服务
    service_process = start_management_service()
    
    if service_process:
        # 测试服务
        test_management_service()
        
        print("\nManagement Model service is running. Press Ctrl+C to stop.")
        try:
            # 保持脚本运行，直到用户中断
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping Management Model service...")
            service_process.terminate()
            service_process.wait(timeout=5)
            print("Management Model service stopped.")
    else:
        print("Failed to start Management Model service.")