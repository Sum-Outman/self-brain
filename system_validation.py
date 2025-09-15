#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self Brain AGI System Validation Script
Comprehensive testing of all models and their functionalities
"""

import requests
import json
import time
import subprocess
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

class SystemValidator:
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.api_endpoints = {
            'main': 'http://localhost:5000',
            'knowledge_manage': 'http://localhost:5000/knowledge_manage',
            'training': 'http://localhost:5000/training',
            'system_settings': 'http://localhost:5000/system_settings',
            'upload': 'http://localhost:5000/upload',
            'a_management': 'http://localhost:5015',
            'i_knowledge': 'http://localhost:5009',
            'k_programming': 'http://localhost:5011',
            'b_language': 'http://localhost:5002',
            'c_audio': 'http://localhost:5003',
            'd_image': 'http://localhost:5004',
            'e_video': 'http://localhost:5005',
            'f_spatial': 'http://localhost:5006',
            'g_sensor': 'http://localhost:5007',
            'h_computer': 'http://localhost:5008',
            'j_motion': 'http://localhost:5010'
        }
        self.results = {}
        
    def test_endpoint(self, name, url, method='GET', data=None):
        """测试单个端点"""
        try:
            if method == 'GET':
                response = requests.get(url, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=data, timeout=10)
            
            if response.status_code == 200:
                self.results[name] = {'status': 'success', 'response': response.text[:200]}
                print(f"✅ {name}: OK")
                return True
            else:
                self.results[name] = {'status': 'error', 'code': response.status_code}
                print(f"❌ {name}: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.results[name] = {'status': 'error', 'error': str(e)}
            print(f"❌ {name}: {str(e)}")
            return False
    
    def start_all_services(self):
        """启动所有必要的服务"""
        print("🚀 Starting all AGI services...")
        
        # 启动A管理模型
        print("Starting A Management Model...")
        subprocess.Popen(['python', 'manager_model/app.py'], 
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        
        # 启动Web界面
        print("Starting Web Interface...")
        subprocess.Popen(['python', 'web_interface/app.py'], 
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        
        # 启动关键子模型
        models_to_start = [
            ('I_knowledge', 'sub_models/I_knowledge/app.py'),
            ('K_programming', 'sub_models/K_programming/programming_api.py'),
            ('D_image', 'sub_models/D_image/app.py'),
            ('G_sensor', 'sub_models/G_sensor/app.py'),
            ('F_spatial', 'sub_models/F_spatial/app.py')
        ]
        
        for name, path in models_to_start:
            if os.path.exists(path):
                print(f"Starting {name}...")
                subprocess.Popen(['python', path], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(1)
    
    def validate_web_pages(self):
        """验证所有网页功能"""
        print("\n🌐 Validating web pages...")
        
        pages = [
            ('main', self.api_endpoints['main']),
            ('knowledge_manage', self.api_endpoints['knowledge_manage']),
            ('training', self.api_endpoints['training']),
            ('system_settings', self.api_endpoints['system_settings']),
            ('upload', self.api_endpoints['upload'])
        ]
        
        for name, url in pages:
            self.test_endpoint(f"page_{name}", url)
    
    def validate_model_apis(self):
        """验证所有模型API"""
        print("\n🔧 Validating model APIs...")
        
        # A管理模型测试
        self.test_endpoint('a_management_health', f"{self.api_endpoints['a_management']}/api/chat", 'POST', 
                          {'message': 'test system status'})
        
        # I知识库模型测试
        self.test_endpoint('i_knowledge_health', f"{self.api_endpoints['i_knowledge']}/health")
        self.test_endpoint('i_knowledge_assist', f"{self.api_endpoints['i_knowledge']}/collaborate", 'POST',
                          {'model': 'D_image', 'task': {'description': 'analyze image content'}})
        
        # K编程模型测试
        self.test_endpoint('k_programming_health', f"{self.api_endpoints['k_programming']}/health")
        
        # D图像模型测试
        self.test_endpoint('d_image_health', f"{self.api_endpoints['d_image']}/health")
        
        # G传感器模型测试
        self.test_endpoint('g_sensor_health', f"{self.api_endpoints['g_sensor']}/health")
        self.test_endpoint('g_sensor_data', f"{self.api_endpoints['g_sensor']}/api/sensors/realtime")
        
        # F空间模型测试
        self.test_endpoint('f_spatial_health', f"{self.api_endpoints['f_spatial']}/health")
    
    def validate_model_functionalities(self):
        """验证模型具体功能"""
        print("\n⚙️ Validating model functionalities...")
        
        # 测试知识库查询
        try:
            response = requests.post(f"{self.api_endpoints['i_knowledge']}/query", 
                                   json={'query': 'What is the speed of light?', 'domain': 'physics'}, 
                                   timeout=10)
            if response.status_code == 200:
                print("✅ Knowledge query: OK")
            else:
                print(f"❌ Knowledge query: {response.status_code}")
        except Exception as e:
            print(f"❌ Knowledge query: {e}")
        
        # 测试传感器数据获取
        try:
            response = requests.get(f"{self.api_endpoints['g_sensor']}/api/sensors/realtime", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'sensors' in data:
                    print("✅ Sensor data: OK")
                else:
                    print("❌ Sensor data: Missing sensors field")
            else:
                print(f"❌ Sensor data: {response.status_code}")
        except Exception as e:
            print(f"❌ Sensor data: {e}")
        
        # 测试编程模型功能
        try:
            response = requests.post(f"{self.api_endpoints['k_programming']}/optimize_code", 
                                   json={'code': 'def test(): pass', 'language': 'python'}, timeout=10)
            if response.status_code == 200:
                print("✅ Programming optimization: OK")
            else:
                print(f"❌ Programming optimization: {response.status_code}")
        except Exception as e:
            print(f"❌ Programming optimization: {e}")
    
    def generate_report(self):
        """生成验证报告"""
        print("\n📊 Generating validation report...")
        
        success_count = sum(1 for r in self.results.values() if r.get('status') == 'success')
        total_count = len(self.results)
        
        report = {
            'total_tests': total_count,
            'successful_tests': success_count,
            'failed_tests': total_count - success_count,
            'success_rate': (success_count / total_count * 100) if total_count > 0 else 0,
            'detailed_results': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存报告
        with open('system_validation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Validation Summary:")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Successful: {report['successful_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Success Rate: {report['success_rate']:.1f}%")
        
        return report
    
    def run_full_validation(self):
        """运行完整验证"""
        print("🎯 Starting Self Brain AGI System Validation...")
        
        # 启动服务
        self.start_all_services()
        time.sleep(5)  # 等待服务完全启动
        
        # 验证所有功能
        self.validate_web_pages()
        self.validate_model_apis()
        self.validate_model_functionalities()
        
        # 生成报告
        return self.generate_report()

if __name__ == "__main__":
    validator = SystemValidator()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # 快速验证（不启动服务）
        validator.validate_web_pages()
        validator.validate_model_apis()
        report = validator.generate_report()
    else:
        # 完整验证
        report = validator.run_full_validation()
    
    print("\n✅ Validation complete! Check system_validation_report.json for details.")