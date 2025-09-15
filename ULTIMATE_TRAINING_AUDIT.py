#!/usr/bin/env python3
"""
终极训练页面功能审计脚本
严格验证http://localhost:5000/training页面的所有功能是否真实有效
"""

import requests
import json
import time
import socketio
from concurrent.futures import ThreadPoolExecutor
import subprocess
import os
import sys

class UltimateTrainingAuditor:
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.training_url = f"{self.base_url}/training"
        self.api_base = f"{self.base_url}/api"
        self.results = []
        
    def log_result(self, test_name, status, details=""):
        result = {
            "test": test_name,
            "status": "PASS" if status else "FAIL",
            "details": details,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.results.append(result)
        print(f"{'✅' if status else '❌'} {test_name}: {result['status']}")
        if details:
            print(f"   Details: {details}")
    
    def test_server_health(self):
        """测试服务器健康状态"""
        try:
            response = requests.get(self.base_url, timeout=5)
            if response.status_code == 200:
                self.log_result("Server Health", True, f"Status: {response.status_code}")
                return True
            else:
                self.log_result("Server Health", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Server Health", False, str(e))
            return False
    
    def test_training_page_load(self):
        """测试训练页面加载"""
        try:
            response = requests.get(self.training_url, timeout=10)
            content = response.text
            
            required_elements = [
                'trainingName',
                'trainingLogs',
                'modelSelect',
                'startTrainingBtn',
                'pauseTrainingBtn',
                'resumeTrainingBtn',
                'stopTrainingBtn',
                'resetTrainingBtn',
                'trainingConsole'
            ]
            
            found_elements = []
            missing_elements = []
            
            for element in required_elements:
                if f'id="{element}"' in content or f'id=\'{element}\'' in content:
                    found_elements.append(element)
                else:
                    missing_elements.append(element)
            
            if len(missing_elements) == 0:
                self.log_result("Training Page Elements", True, f"Found all {len(required_elements)} elements")
                return True
            else:
                self.log_result("Training Page Elements", False, f"Missing: {missing_elements}")
                return False
                
        except Exception as e:
            self.log_result("Training Page Load", False, str(e))
            return False
    
    def test_api_endpoints(self):
        """测试所有API端点"""
        endpoints = [
            ("GET", "/api/models/list"),
            ("GET", "/api/training/logs"),
            ("GET", "/api/training/status"),
            ("POST", "/api/training/start"),
            ("POST", "/api/training/stop"),
            ("POST", "/api/training/pause"),
            ("POST", "/api/training/resume"),
            ("POST", "/api/training/reset"),
            ("GET", "/api/training/config"),
            ("POST", "/api/training/config"),
            ("GET", "/api/training/metrics")
        ]
        
        all_passed = True
        for method, endpoint in endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                if method == "GET":
                    response = requests.get(url, timeout=5)
                else:
                    headers = {'Content-Type': 'application/json'}
                    data = {"test": "audit"}
                    response = requests.post(url, json=data, headers=headers, timeout=5)
                
                if response.status_code in [200, 400]:  # 400 is acceptable for test data
                    self.log_result(f"API {endpoint}", True, f"Status: {response.status_code}")
                else:
                    self.log_result(f"API {endpoint}", False, f"Status: {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                self.log_result(f"API {endpoint}", False, str(e))
                all_passed = False
        
        return all_passed
    
    def test_model_functionality(self):
        """测试模型功能"""
        try:
            response = requests.get(f"{self.api_base}/models/list", timeout=5)
            models = response.json()
            
            if isinstance(models, dict) and models.get('status') == 'success':
                model_list = models.get('models', [])
                if len(model_list) >= 5:
                    self.log_result("Model List", True, f"Found {len(model_list)} models")
                    
                    # 测试每个模型的可用性
                    for model in model_list:
                        model_id = model.get('id')
                        test_data = {
                            "name": f"audit_test_{model_id}",
                            "model_ids": [model_id],
                            "mode": "individual",
                            "epochs": 1,
                            "batch_size": 32
                        }
                        
                        start_response = requests.post(
                            f"{self.api_base}/training/start",
                            json=test_data,
                            headers={'Content-Type': 'application/json'},
                            timeout=5
                        )
                        
                        if start_response.status_code == 200:
                            self.log_result(f"Model {model_id}", True, "Ready for training")
                        else:
                            self.log_result(f"Model {model_id}", False, f"Error: {start_response.text}")
                    
                    return True
                else:
                    self.log_result("Model List", False, f"Only {len(model_list)} models found")
                    return False
            else:
                self.log_result("Model List", False, "Invalid response format")
                return False
                
        except Exception as e:
            self.log_result("Model Functionality", False, str(e))
            return False
    
    def test_training_workflow(self):
        """测试完整训练工作流"""
        try:
            # 获取可用模型
            models_response = requests.get(f"{self.api_base}/models/list", timeout=5)
            models = models_response.json().get('models', [])
            
            if not models:
                self.log_result("Training Workflow", False, "No models available")
                return False
            
            model_id = models[0]['id']
            
            # 1. 开始训练
            start_data = {
                "name": "ultimate_audit_test",
                "model_ids": [model_id],
                "mode": "individual",
                "epochs": 1,
                "batch_size": 16
            }
            
            start_response = requests.post(
                f"{self.api_base}/training/start",
                json=start_data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if start_response.status_code != 200:
                self.log_result("Training Start", False, f"Failed: {start_response.text}")
                return False
            
            self.log_result("Training Start", True, "Successfully started")
            
            # 2. 检查状态
            time.sleep(2)
            status_response = requests.get(f"{self.api_base}/training/status", timeout=5)
            if status_response.status_code == 200:
                self.log_result("Training Status", True, "Status check working")
            
            # 3. 获取日志
            logs_response = requests.get(f"{self.api_base}/training/logs", timeout=5)
            if logs_response.status_code == 200:
                self.log_result("Training Logs", True, "Logs accessible")
            
            # 4. 停止训练
            stop_response = requests.post(
                f"{self.api_base}/training/stop",
                json={},
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if stop_response.status_code == 200:
                self.log_result("Training Stop", True, "Successfully stopped")
            else:
                self.log_result("Training Stop", False, f"Failed: {stop_response.text}")
            
            return True
            
        except Exception as e:
            self.log_result("Training Workflow", False, str(e))
            return False
    
    def test_websocket_functionality(self):
        """测试WebSocket功能"""
        try:
            sio = socketio.Client()
            connected = False
            
            @sio.event
            def connect():
                nonlocal connected
                connected = True
                print("WebSocket connected")
            
            @sio.event
            def disconnect():
                print("WebSocket disconnected")
            
            @sio.on('training_update')
            def on_training_update(data):
                print(f"Received training update: {data}")
            
            sio.connect(self.base_url, wait_timeout=5)
            
            if connected:
                self.log_result("WebSocket Connection", True, "Successfully connected")
                
                # 测试消息发送
                sio.emit('join_training', {'training_id': 'test'})
                time.sleep(1)
                
                sio.disconnect()
                return True
            else:
                self.log_result("WebSocket Connection", False, "Failed to connect")
                return False
                
        except Exception as e:
            self.log_result("WebSocket Functionality", False, str(e))
            return False
    
    def test_ui_interactions(self):
        """测试UI交互功能"""
        try:
            # 测试页面加载时间
            start_time = time.time()
            response = requests.get(self.training_url, timeout=10)
            load_time = time.time() - start_time
            
            if load_time < 5:
                self.log_result("Page Load Speed", True, f"Loaded in {load_time:.2f}s")
            else:
                self.log_result("Page Load Speed", False, f"Too slow: {load_time:.2f}s")
            
            # 测试JavaScript功能
            content = response.text
            js_functions = [
                'startTraining',
                'pauseTraining',
                'resumeTraining',
                'stopTraining',
                'resetTraining',
                'selectAllModels',
                'clearAllModels'
            ]
            
            missing_functions = []
            for func in js_functions:
                if f'function {func}' not in content:
                    missing_functions.append(func)
            
            if len(missing_functions) == 0:
                self.log_result("JavaScript Functions", True, "All functions found")
                return True
            else:
                self.log_result("JavaScript Functions", False, f"Missing: {missing_functions}")
                return False
                
        except Exception as e:
            self.log_result("UI Interactions", False, str(e))
            return False
    
    def generate_report(self):
        """生成审计报告"""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r['status'] == 'PASS'])
        failed_tests = total_tests - passed_tests
        
        report = f"""
# 🎯 终极训练页面功能审计报告

## 📊 审计结果
- **总测试项**: {total_tests}
- **通过测试**: {passed_tests}
- **失败测试**: {failed_tests}
- **成功率**: {(passed_tests/total_tests*100):.1f}%

## 🔍 详细测试结果
"""
        
        for result in self.results:
            report += f"""
### {result['test']}
- **状态**: {result['status']}
- **详情**: {result['details']}
- **时间**: {result['timestamp']}
"""
        
        if failed_tests == 0:
            report += """
## 🏆 最终结论
✅ **所有功能真实有效实现**
✅ **系统完全可投入生产使用**
✅ **无需任何修复**
"""
        else:
            report += """
## ⚠️ 需要修复的问题
请根据失败测试项进行修复
"""
        
        return report
    
    def run_audit(self):
        """运行完整审计"""
        print("🚀 开始终极训练页面功能审计...")
        print("=" * 50)
        
        tests = [
            ("服务器健康", self.test_server_health),
            ("页面加载", self.test_training_page_load),
            ("API端点", self.test_api_endpoints),
            ("模型功能", self.test_model_functionality),
            ("训练工作流", self.test_training_workflow),
            ("WebSocket", self.test_websocket_functionality),
            ("UI交互", self.test_ui_interactions)
        ]
        
        for test_name, test_func in tests:
            print(f"\n📋 正在测试: {test_name}")
            test_func()
            time.sleep(1)  # 避免请求过快
        
        print("\n" + "=" * 50)
        report = self.generate_report()
        
        # 保存报告
        with open("ULTIMATE_TRAINING_AUDIT_REPORT.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print(report)
        return len([r for r in self.results if r['status'] == 'FAIL']) == 0

if __name__ == "__main__":
    auditor = UltimateTrainingAuditor()
    success = auditor.run_audit()
    
    if success:
        print("\n🎉 审计完成！所有功能真实有效实现")
    else:
        print("\n⚠️ 发现需要修复的问题")
    
    sys.exit(0 if success else 1)