#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final System Status Check
"""

import requests
import json
import time
import socket

def check_port(host, port):
    """检查端口是否开放"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def test_endpoint(name, url, expected_status=200):
    """测试HTTP端点"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == expected_status:
            print(f"✅ {name}: OK")
            return True
        else:
            print(f"❌ {name}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name}: {str(e)}")
        return False

def main():
    print("🔍 Final System Status Check")
    print("=" * 50)
    
    # 检查端口开放情况
    ports = {
        5000: "Web Interface",
        5015: "A Management Model", 
        5009: "I Knowledge Expert",
        5011: "K Programming Model",
        5004: "D Image Processing",
        5007: "G Sensor Perception",
        5006: "F Spatial Location",
        5003: "C Audio Processing",
        5002: "B Language Model",
        5005: "E Video Processing",
        5008: "H Computer Control",
        5010: "J Motion Control"
    }
    
    print("\n📡 Port Status:")
    open_ports = []
    for port, name in ports.items():
        if check_port('localhost', port):
            print(f"✅ {port} ({name}): OPEN")
            open_ports.append(port)
        else:
            print(f"❌ {port} ({name}): CLOSED")
    
    print(f"\n📊 Summary: {len(open_ports)}/{len(ports)} ports open")
    
    # 测试关键端点的HTTP响应
    print("\n🌐 HTTP Endpoints:")
    endpoints = [
        ("Web Interface", "http://localhost:5000"),
        ("Knowledge Management", "http://localhost:5000/knowledge_manage"),
        ("Training Page", "http://localhost:5000/training"),
        ("System Settings", "http://localhost:5000/system_settings"),
        ("Upload Page", "http://localhost:5000/upload"),
        ("A Management API", "http://localhost:5015/health"),
        ("I Knowledge API", "http://localhost:5009/health"),
        ("K Programming API", "http://localhost:5011/health"),
        ("D Image API", "http://localhost:5004/health"),
        ("G Sensor API", "http://localhost:5007/health"),
        ("F Spatial API", "http://localhost:5006/health"),
    ]
    
    working_endpoints = 0
    for name, url in endpoints:
        if test_endpoint(name, url):
            working_endpoints += 1
    
    print(f"\n📈 HTTP Summary: {working_endpoints}/{len(endpoints)} endpoints working")
    
    # 生成最终报告
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'open_ports': open_ports,
        'total_ports': len(ports),
        'working_endpoints': working_endpoints,
        'total_endpoints': len(endpoints),
        'system_status': 'operational' if len(open_ports) >= 8 else 'partial',
        'coverage': f"{len(open_ports)}/{len(ports)} models active"
    }
    
    with open('final_system_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n🎯 Final Status: {report['system_status'].upper()}")
    print(f"📊 Coverage: {report['coverage']}")
    print("📋 Report saved to: final_system_report.json")

if __name__ == "__main__":
    main()