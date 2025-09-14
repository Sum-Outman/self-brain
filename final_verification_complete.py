#!/usr/bin/env python3
"""
知识管理系统最终验证脚本
验证所有核心功能的完整性和正确性
"""

import requests
import json
import sys
import os
from datetime import datetime

def test_endpoint(url, method='GET', data=None, expected_status=200):
    """测试API端点"""
    try:
        if method == 'GET':
            response = requests.get(url)
        elif method == 'POST':
            response = requests.post(url, json=data)
        elif method == 'DELETE':
            response = requests.delete(url)
        
        success = response.status_code == expected_status
        return {
            'url': url,
            'method': method,
            'status_code': response.status_code,
            'success': success,
            'response': response.json() if response.content else None
        }
    except Exception as e:
        return {
            'url': url,
            'method': method,
            'status_code': 0,
            'success': False,
            'error': str(e)
        }

def main():
    base_url = "http://localhost:5000"
    tests = []
    
    print("🧪 知识管理系统最终验证测试")
    print("=" * 50)
    
    # 1. 测试知识管理页面
    print("\n📋 1. 测试知识管理页面...")
    result = test_endpoint(f"{base_url}/knowledge_manage")
    tests.append({
        'name': '知识管理页面访问',
        'result': result
    })
    print(f"   ✅ 状态: {'通过' if result['success'] else '失败'}")
    
    # 2. 测试知识列表API
    print("\n📚 2. 测试知识列表API...")
    result = test_endpoint(f"{base_url}/api/knowledge")
    tests.append({
        'name': '知识列表API',
        'result': result
    })
    
    if result['success'] and result['response']:
        entries = result['response'].get('entries', [])
        print(f"   ✅ 获取条目数: {len(entries)}")
        
        # 3. 测试单个知识查看
        if entries:
            test_id = entries[0]['id']
            print(f"\n👁️  3. 测试知识查看页面...")
            result = test_endpoint(f"{base_url}/knowledge/view/{test_id}")
            tests.append({
                'name': '知识查看页面',
                'result': result
            })
            print(f"   ✅ 状态: {'通过' if result['success'] else '失败'}")
            
            # 4. 测试知识详情API
            print(f"\n📖 4. 测试知识详情API...")
            result = test_endpoint(f"{base_url}/api/knowledge/detail/{test_id}")
            tests.append({
                'name': '知识详情API',
                'result': result
            })
            print(f"   ✅ 状态: {'通过' if result['success'] else '失败'}")
    
    # 5. 测试知识统计API
    print("\n📊 5. 测试知识统计API...")
    result = test_endpoint(f"{base_url}/api/knowledge/stats")
    tests.append({
        'name': '知识统计API',
        'result': result
    })
    if result['success'] and result['response']:
        stats = result['response'].get('stats', {})
        print(f"   ✅ 总条目: {stats.get('total_count', 0)}")
    
    # 6. 测试搜索功能
    print("\n🔍 6. 测试搜索功能...")
    result = test_endpoint(f"{base_url}/api/knowledge?search=test")
    tests.append({
        'name': '搜索功能',
        'result': result
    })
    print(f"   ✅ 状态: {'通过' if result['success'] else '失败'}")
    
    # 7. 测试分类筛选
    print("\n🏷️  7. 测试分类筛选...")
    result = test_endpoint(f"{base_url}/api/knowledge?category=General")
    tests.append({
        'name': '分类筛选',
        'result': result
    })
    print(f"   ✅ 状态: {'通过' if result['success'] else '失败'}")
    
    # 8. 测试批量删除API
    print("\n🗑️  8. 测试批量删除API...")
    result = test_endpoint(f"{base_url}/api/knowledge/delete_selected", 
                          method='POST', 
                          data={'ids': []})
    tests.append({
        'name': '批量删除API',
        'result': result
    })
    print(f"   ✅ 状态: {'通过' if result['success'] else '失败'}")
    
    # 9. 测试单个删除API
    if entries:
        test_id = entries[0]['id']
        print(f"\n🗑️  9. 测试单个删除API...")
        result = test_endpoint(f"{base_url}/api/knowledge/{test_id}", method='DELETE')
        tests.append({
            'name': '单个删除API',
            'result': result
        })
        print(f"   ✅ 状态: {'通过' if result['success'] else '失败'}")
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总")
    print("=" * 50)
    
    total_tests = len(tests)
    passed_tests = sum(1 for t in tests if t['result']['success'])
    
    for test in tests:
        status = "✅ 通过" if test['result']['success'] else "❌ 失败"
        print(f"{status} {test['name']}")
    
    print(f"\n🎯 总计: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("🎉 所有功能验证通过！系统运行正常")
        return True
    else:
        print("⚠️  部分功能存在问题，需要检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)