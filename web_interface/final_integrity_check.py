#!/usr/bin/env python3
"""Final Integrity Verification - Check the actual effect of each page and function"""
import requests
import json
import os
import tempfile
from datetime import datetime

BASE_URL = "http://localhost:5000"

class FinalIntegrityChecker:
    def __init__(self):
        self.session = requests.Session()
        self.results = {}
        
    def run_complete_check(self):
        """Run complete check"""
        print("🔍 Starting final integrity verification...")
        
        # Check each function of each page
        self.check_all_pages()
        self.check_all_apis()
        self.check_file_operations()
        self.check_real_time_features()
        self.check_error_handling()
        
        self.generate_report()
    
    def check_all_pages(self):
        """Check all page functions"""
        pages = [
            ('/', 'Home'),
            ('/dashboard', 'Dashboard'),
            ('/ai_chat', 'AI Chat'),
            ('/advanced_chat', 'Advanced Chat'),
            ('/knowledge_base', 'Knowledge Base'),
            ('/model_management', 'Model Management'),
            ('/training_center', 'Training Center'),
            ('/system_settings', 'System Settings'),
            ('/help', 'Help'),
            ('/license', 'License'),
            ('/about', 'About'),
            ('/system_status', 'System Status')
        ]
        
        print("\n📄 Page access verification:")
        for path, name in pages:
            try:
                r = self.session.get(f"{BASE_URL}{path}")
                self.results[f"page_{path}"] = r.status_code == 200
                status = "✅" if r.status_code == 200 else "❌"
                print(f"{status} {name} ({path}): {r.status_code}")
            except Exception as e:
                self.results[f"page_{path}"] = False
                print(f"❌ {name} ({path}): {e}")
    
    def check_all_apis(self):
        """Check all API functions"""
        apis = [
            ('GET', '/api/system/status', 'System Status API'),
            ('GET', '/api/models', 'Model List API'),
            ('GET', '/api/knowledge/list', 'Knowledge Base List API'),
            ('GET', '/api/training/status', 'Training Status API'),
            ('GET', '/api/dashboard/data', 'Dashboard Data API'),
            ('GET', '/api/settings/load', 'Settings Load API'),
            ('GET', '/api/system/resources', 'System Resources API')
        ]
        
        print("\n🔌 API function verification:")
        for method, path, name in apis:
            try:
                if method == 'GET':
                    r = self.session.get(f"{BASE_URL}{path}")
                else:
                    r = self.session.post(f"{BASE_URL}{path}")
                
                success = r.status_code in [200, 201]
                self.results[f"api_{path}"] = success
                status = "✅" if success else "❌"
                print(f"{status} {name}: {r.status_code}")
                
            except Exception as e:
                self.results[f"api_{path}"] = False
                print(f"❌ {name}: {e}")
    
    def check_file_operations(self):
        """Check file operation functions"""
        print("\n📁 File operation verification:")
        
        # 1. Test file upload
        try:
            test_content = "Test file content - " + str(datetime.now())
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                temp_path = f.name
            
            with open(temp_path, 'rb') as f:
                files = {'file': f}
                r = self.session.post(f"{BASE_URL}/api/knowledge/upload", files=files)
            
            upload_success = r.status_code == 200
            self.results['file_upload'] = upload_success
            print(f"{'✅' if upload_success else '❌'} File upload: {r.status_code}")
            
            os.unlink(temp_path)
            
        except Exception as e:
            self.results['file_upload'] = False
            print(f"❌ File upload: {e}")
        
        # 2. Test file download/read
        try:
            r = self.session.get(f"{BASE_URL}/api/knowledge/list")
            knowledge = r.json().get('knowledge', [])
            self.results['file_list'] = len(knowledge) > 0
            print(f"{'✅' if len(knowledge) > 0 else '❌'} File list: {len(knowledge)} files")
        except Exception as e:
            self.results['file_list'] = False
            print(f"❌ File list: {e}")
    
    def check_real_time_features(self):
        """Check real-time features"""
        print("\n⚡ Real-time feature verification:")
        
        # 1. Check WebSocket connection
        try:
            r = self.session.get(f"{BASE_URL}/socket.io/")
            socket_success = r.status_code == 200
            self.results['websocket'] = socket_success
            print(f"{'✅' if socket_success else '❌'} WebSocket connection: {r.status_code}")
        except Exception as e:
            self.results['websocket'] = False
            print(f"❌ WebSocket connection: {e}")
        
        # 2. Check real-time data update
        try:
            r = self.session.get(f"{BASE_URL}/api/system/resources")
            resources = r.json()
            real_time_success = 'cpu' in resources or 'memory' in resources
            self.results['real_time_data'] = real_time_success
            print(f"{'✅' if real_time_success else '⚠️'} Real-time data: {'Available' if real_time_success else 'Partially unavailable'}")
        except Exception as e:
            self.results['real_time_data'] = False
            print(f"❌ Real-time data: {e}")
    
    def check_error_handling(self):
        """Check error handling mechanism"""
        print("\n🛡️ Error handling verification:")
        
        # 1. Test invalid API call
        try:
            r = self.session.get(f"{BASE_URL}/api/nonexistent")
            handled = r.status_code == 404
            self.results['error_404'] = handled
            print(f"{'✅' if handled else '❌'} 404 error handling: {r.status_code}")
        except Exception as e:
            self.results['error_404'] = False
            print(f"❌ 404 error handling: {e}")
        
        # 2. Test invalid data submission
        try:
            r = self.session.post(f"{BASE_URL}/api/settings/general", json={"invalid": "data"})
            handled = r.status_code in [400, 422]
            self.results['error_invalid'] = handled
            print(f"{'✅' if handled else '❌'} Invalid data handling: {r.status_code}")
        except Exception as e:
            self.results['error_invalid'] = False
            print(f"❌ Invalid data handling: {e}")
    
    def generate_report(self):
        """Generate final report"""
        print("\n" + "="*50)
        print("📊 Final Integrity Verification Report")
        print("="*50)
        
        total_checks = len(self.results)
        passed_checks = sum(1 for v in self.results.values() if v)
        
        print(f"Total checks: {total_checks}")
        print(f"Passed checks: {passed_checks}")
        print(f"Failed checks: {total_checks - passed_checks}")
        print(f"Pass rate: {(passed_checks/total_checks)*100:.1f}%")
        
        print("\nDetailed results:")
        for check, passed in self.results.items():
            status = "✅" if passed else "❌"
            print(f"{status} {check}")
        
        # Critical functions confirmation
        critical_functions = [
            'page_/',
            'page_/dashboard',
            'page_/ai_chat',
            'page_/knowledge_base',
            'api_/api/system/status',
            'api_/api/models',
            'file_upload',
            'real_time_data'
        ]
        
        critical_passed = all(self.results.get(f, False) for f in critical_functions)
        
        print(f"\n🎯 Critical functions status: {'All normal' if critical_passed else 'Partial abnormal'}")
        print("="*50)

if __name__ == "__main__":
    checker = FinalIntegrityChecker()
    checker.run_complete_check()
