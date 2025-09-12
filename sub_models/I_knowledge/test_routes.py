#!/usr/bin/env python3
"""
测试所有知识库系统路由是否正常工作的脚本
"""

import requests
import sys

def test_routes():
    """测试所有路由"""
    base_url = "http://localhost:8003"
    
    routes = [
        "/",  # 主页
        "/dashboard",  # 控制面板
        "/import",  # 知识导入
        "/knowledge",  # 知识管理
        "/chat",  # AI对话
        "/analytics",  # 数据分析
        "/settings",  # 系统设置
        "/help",  # 帮助中心
        "/profile",  # 个人资料
        "/preferences",  # 偏好设置
        "/knowledge_interface",  # 知识库专家界面
    ]
    
    print("测试知识库系统所有路由...")
    print("=" * 50)
    
    all_success = True
    
    for route in routes:
        try:
            response = requests.get(f"{base_url}{route}", timeout=5)
            status = "✅ 正常" if response.status_code == 200 else f"❌ 错误 ({response.status_code})"
            print(f"{route:<20} {status}")
            
            if response.status_code != 200:
                all_success = False
                
        except requests.exceptions.RequestException as e:
            print(f"{route:<20} ❌ 连接失败: {e}")
            all_success = False
    
    print("=" * 50)
    if all_success:
        print("🎉 所有路由测试通过！")
    else:
        print("⚠️  部分路由存在问题，请检查应用日志")
    
    return all_success

if __name__ == "__main__":
    success = test_routes()
    sys.exit(0 if success else 1)