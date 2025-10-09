#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self Brain AGI System - Port Checker
检查所有服务端口是否可用
"""

import socket
import requests
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

class PortChecker:
    def __init__(self):
        self.services = {
            "Main Web Interface": {"port": 8080, "url": "http://localhost:8080"},
            "A Management Model": {"port": 5000, "url": "http://localhost:5000/api/health"},
            "B Language Model": {"port": 5001, "url": "http://localhost:5001/api/health"},
            "C Audio Model": {"port": 5002, "url": "http://localhost:5002/api/health"},
            "D Image Model": {"port": 5003, "url": "http://localhost:5003/api/health"},
            "E Video Model": {"port": 5004, "url": "http://localhost:5004/api/health"},
            "F Spatial Model": {"port": 5005, "url": "http://localhost:5005/api/health"},
            "G Sensor Model": {"port": 5006, "url": "http://localhost:5006/api/health"},
            "H Computer Control": {"port": 5007, "url": "http://localhost:5007/api/health"},
            "I Knowledge Model": {"port": 5008, "url": "http://localhost:5008/api/health"},
            "J Motion Model": {"port": 5009, "url": "http://localhost:5009/api/health"},
            "K Programming Model": {"port": 5010, "url": "http://localhost:5010/api/health"},
            "Training Manager": {"port": 5012, "url": "http://localhost:5012/api/health"},
            "AGI Core": {"port": 5014, "url": "http://localhost:5014/api/health"},
            "Manager Model API": {"port": 5015, "url": "http://localhost:5015/api/health"}
        }
    
    def check_port(self, port):
        """检查端口是否可用"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return port, result == 0
        except Exception as e:
            return port, False
    
    def check_service_health(self, name, service):
        """检查服务健康状态"""
        try:
            response = requests.get(service["url"], timeout=2)
            return name, response.status_code == 200
        except requests.exceptions.RequestException:
            return name, False
    
    def run_checks(self):
        """运行所有检查"""
        print("🔍 Self Brain AGI System - Port Status Check")
        print("=" * 50)
        
        # 检查端口占用
        print("\n📡 检查端口占用...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            port_futures = {executor.submit(self.check_port, service["port"]): name 
                           for name, service in self.services.items()}
            
            port_results = {}
            for future in as_completed(port_futures):
                name = port_futures[future]
                port, is_occupied = future.result()
                port_results[name] = (port, is_occupied)
                status = "🔴 占用" if is_occupied else "🟢 可用"
                print(f"  {name}: Port {port} - {status}")
        
        # 检查服务健康
        print("\n🏥 检查服务健康...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            health_futures = {executor.submit(self.check_service_health, name, service): name 
                            for name, service in self.services.items()}
            
            health_results = {}
            for future in as_completed(health_futures):
                name = health_futures[future]
                service_name, is_healthy = future.result()
                health_results[service_name] = is_healthy
                
                port, is_occupied = port_results[name]
                if is_occupied:
                    status = "🟢 健康" if is_healthy else "🔴 不健康"
                    print(f"  {name}: {status}")
                else:
                    print(f"  {name}: ⚪ 未启动")
        
        # 总结
        print("\n📊 总结:")
        total_services = len(self.services)
        occupied_ports = sum(1 for _, is_occupied in port_results.values() if is_occupied)
        healthy_services = sum(1 for is_healthy in health_results.values() if is_healthy)
        
        print(f"  总服务数: {total_services}")
        print(f"  已占用端口: {occupied_ports}")
        print(f"  健康服务: {healthy_services}")
        
        return {
            "total": total_services,
            "occupied": occupied_ports,
            "healthy": healthy_services,
            "port_results": port_results,
            "health_results": health_results
        }

if __name__ == "__main__":
    checker = PortChecker()
    results = checker.run_checks()
    
    print("\n" + "=" * 50)
    print("🎯 启动建议:")
    
    if results["occupied"] == 0:
        print("✅ 所有端口可用，可以安全启动系统")
    else:
        print("⚠️  部分端口已被占用，建议检查冲突")
    
    print("\n📋 启动命令:")
    print("  python start_system_updated.bat")
    print("  python a_manager_standalone.py")
    print("  cd manager_model && python app.py")