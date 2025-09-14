#!/usr/bin/env python3
"""
API端点审计脚本
检查知识管理系统的所有实际API端点
"""

import requests
import json
import time

class APIAuditor:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def audit_all_endpoints(self):
        """审计所有知识相关的API端点"""
        
        # 实际存在的端点列表
        endpoints = [
            # 基础CRUD
            ("GET", "/api/knowledge"),
            ("GET", "/api/knowledge/list"),
            ("POST", "/api/knowledge/create"),
            ("GET", "/api/knowledge/<id>"),
            ("PUT", "/api/knowledge/<id>"),
            ("DELETE", "/api/knowledge/<id>"),
            
            # 批量操作
            ("POST", "/api/knowledge/delete_selected"),
            ("POST", "/api/knowledge/export_selected"),
            
            # 搜索和筛选
            ("GET", "/api/knowledge/search"),
            ("GET", "/api/knowledge/search/enhanced"),
            ("GET", "/api/knowledge/categories"),
            ("GET", "/api/knowledge/tags"),
            
            # 统计和管理
            ("GET", "/api/knowledge/stats"),
            ("POST", "/api/knowledge/optimize"),
            ("POST", "/api/knowledge/cleanup"),
            
            # 文件操作
            ("POST", "/api/knowledge/upload"),
            ("POST", "/api/knowledge/import/zip"),
            ("POST", "/api/knowledge/import/json"),
            
            # 备份恢复
            ("GET", "/api/knowledge/backup/status"),
            ("POST", "/api/knowledge/restore/<filename>"),
            ("POST", "/api/knowledge/restore"),
            
            # 元数据
            ("GET", "/api/knowledge/metadata/<metadata_id>"),
            ("PUT", "/api/knowledge/metadata/<metadata_id>"),
            
            # 不存在的端点（需要检查）
            ("GET", "/api/knowledge/storage_config"),
            ("GET", "/api/knowledge/storage/info"),
        ]
        
        print("🔍 开始API端点审计...")
        print("=" * 60)
        
        working_endpoints = []
        missing_endpoints = []
        
        for method, path in endpoints:
            # 替换占位符
            if "<id>" in path:
                test_path = path.replace("<id>", "test123")
            elif "<filename>" in path:
                test_path = path.replace("<filename>", "test.json")
            elif "<metadata_id>" in path:
                test_path = path.replace("<metadata_id>", "meta123")
            else:
                test_path = path
                
            url = f"{self.base_url}{test_path}"
            
            try:
                response = self.session.request(method, url, timeout=5)
                
                if response.status_code == 404:
                    missing_endpoints.append(f"{method} {path}")
                    print(f"❌ {method} {path} - 404 Not Found")
                elif response.status_code == 405:
                    # 方法不允许，但端点存在
                    working_endpoints.append(f"{method} {path}")
                    print(f"✅ {method} {path} - 405 Method Allowed (端点存在)")
                else:
                    working_endpoints.append(f"{method} {path}")
                    print(f"✅ {method} {path} - {response.status_code} OK")
                    
            except Exception as e:
                missing_endpoints.append(f"{method} {path}")
                print(f"❌ {method} {path} - 错误: {e}")
        
        print("\n" + "=" * 60)
        print("📊 审计结果:")
        print(f"✅ 工作端点: {len(working_endpoints)}")
        print(f"❌ 缺失端点: {len(missing_endpoints)}")
        
        if missing_endpoints:
            print("\n需要完善的端点:")
            for endpoint in missing_endpoints:
                print(f"  - {endpoint}")
        
        return working_endpoints, missing_endpoints

if __name__ == "__main__":
    auditor = APIAuditor()
    auditor.audit_all_endpoints()