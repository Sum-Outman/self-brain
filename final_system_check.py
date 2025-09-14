#!/usr/bin/env python3
# Copyright 2025 The AI Management System Authors

"""
Self Brain AGI System - 最终系统检查与启动
验证所有端口配置并启动完整系统
"""

import os
import sys
import subprocess
import time
import requests
from concurrent.futures import ThreadPoolExecutor

# 标准端口配置
STANDARD_PORTS = {
    5000: "Main Web Interface",
    5001: "A Management Model",
    5002: "B Language Model",
    5003: "C Audio Model",
    5004: "D Image Model",
    5005: "E Video Model",
    5006: "F Spatial Model",
    5007: "G Sensor Model",
    5008: "H Computer Control",
    5009: "I Knowledge Model",
    5010: "J Motion Model",
    5011: "K Programming Model",
    5012: "Training Manager",
    5013: "Quantum Integration",
    5014: "Standalone A Manager",
    5015: "Manager Model API",
}

# 服务启动命令
SERVICE_COMMANDS = {
    5000: ["python", "web_interface/working_enhanced_chat.py"],
    5001: ["python", "a_management_server.py"],
    5002: ["python", "sub_models/B_language/app.py"],
    5003: ["python", "sub_models/C_audio/api.py"],
    5004: ["python", "sub_models/D_image/api.py"],
    5005: ["python", "sub_models/E_video/api.py"],
    5006: ["python", "sub_models/F_spatial/api.py"],
    5007: ["python", "sub_models/G_sensor/api.py"],
    5008: ["python", "sub_models/H_computer_control/api.py"],
    5009: ["python", "sub_models/I_knowledge/api.py"],
    5010: ["python", "sub_models/J_motion/api.py"],
    5011: ["python", "sub_models/K_programming/programming_api.py"],
    5012: ["python", "training_manager.py"],
    5013: ["python", "quantum_integration.py"],
    5014: ["python", "a_manager_standalone.py"],
    5015: ["python", "manager_model/app.py"],
}

def check_port_health(port):
    """检查端口健康状态"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_service(port):
    """启动单个服务"""
    try:
        cmd = SERVICE_COMMANDS[port]
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)  # 等待服务启动
        
        if check_port_health(port):
            return f"✅ {STANDARD_PORTS[port]} (端口 {port}) - 已启动"
        else:
            return f"❌ {STANDARD_PORTS[port]} (端口 {port}) - 启动失败"
    except Exception as e:
        return f"❌ {STANDARD_PORTS[port]} (端口 {port}) - 错误: {str(e)}"

def main():
    """主函数"""
    print("🚀 Self Brain AGI System - 最终系统检查")
    print("=" * 60)
    
    # 检查端口占用情况
    print("\n📡 检查端口状态...")
    running_services = []
    for port, service_name in STANDARD_PORTS.items():
        if check_port_health(port):
            running_services.append(port)
            print(f"✅ {service_name} (端口 {port}) - 运行中")
        else:
            print(f"⚪ {service_name} (端口 {port}) - 未启动")
    
    # 启动缺失的服务
    missing_services = [p for p in STANDARD_PORTS.keys() if p not in running_services]
    
    if missing_services:
        print(f"\n🔧 启动 {len(missing_services)} 个缺失的服务...")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(start_service, missing_services))
        
        for result in results:
            print(result)
    
    # 最终状态报告
    print("\n📊 最终状态报告:")
    print("=" * 60)
    
    total_services = len(STANDARD_PORTS)
    running_count = len([p for p in STANDARD_PORTS.keys() if check_port_health(p)])
    
    print(f"总服务数: {total_services}")
    print(f"运行中服务: {running_count}")
    print(f"启动成功率: {(running_count/total_services)*100:.1f}%")
    
    if running_count == total_services:
        print("\n🎉 所有服务已成功启动！")
        print("\n访问地址:")
        print("- 主界面: http://localhost:5000")
        print("- 管理界面: http://localhost:5015")
    else:
        print(f"\n⚠️  {total_services - running_count} 个服务启动失败，请检查日志")

if __name__ == "__main__":
    main()