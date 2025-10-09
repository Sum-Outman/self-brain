#!/usr/bin/env python3
"""
修复GPU检测问题 - 确保GPU字段始终存在
Fix GPU detection issue - Ensure GPU fields are always present
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'training_manager'))

from training_manager.advanced_train_control import AdvancedTrainingController
import subprocess
import time

def test_gpu_import():
    """测试GPU库导入"""
    print("=== 测试GPU库导入 ===")
    
    try:
        import GPUtil
        print("✅ GPUtil库导入成功")
        
        # 测试GPU检测
        gpus = GPUtil.getGPUs()
        if gpus:
            print(f"✅ 检测到 {len(gpus)} 个GPU")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i+1}: {gpu.name}, 使用率: {gpu.load*100:.1f}%")
        else:
            print("⚠️  未检测到GPU，将使用模拟数据")
            
    except ImportError as e:
        print(f"❌ GPUtil库导入失败: {e}")
        print("   将使用模拟数据")
    except Exception as e:
        print(f"❌ GPU检测异常: {e}")
        print("   将使用模拟数据")

def restart_flask_app():
    """重启Flask应用"""
    print("\n=== 重启Flask应用 ===")
    
    try:
        # 查找并终止现有Flask进程
        import psutil
        
        flask_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'app.py' in cmdline and 'web_interface' in cmdline:
                    flask_processes.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if flask_processes:
            print(f"找到 {len(flask_processes)} 个Flask进程，正在终止...")
            for pid in flask_processes:
                try:
                    os.kill(pid, 9)
                    print(f"   已终止进程 {pid}")
                except ProcessLookupError:
                    print(f"   进程 {pid} 已不存在")
        
        # 等待片刻
        time.sleep(2)
        
        # 启动新的Flask应用
        print("启动新的Flask应用...")
        subprocess.Popen([sys.executable, 'web_interface/app.py'], 
                        cwd=os.path.dirname(os.path.abspath(__file__)))
        
        print("✅ Flask应用已重启")
        print("   等待10秒让应用完全启动...")
        time.sleep(10)
        
    except Exception as e:
        print(f"重启失败: {e}")

def verify_fix():
    """验证修复效果"""
    print("\n=== 验证修复效果 ===")
    
    try:
        import requests
        response = requests.get('http://localhost:5000/api/system/resources', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            resources = data.get('resources', {})
            
            if 'system' in resources:
                system_data = resources['system']
                
                # 检查GPU字段
                has_gpu_usage = 'gpu_usage_percent' in system_data
                has_gpu_model = 'gpu_model' in system_data
                
                print(f"✅ API响应成功")
                print(f"   system字段键: {list(system_data.keys())}")
                print(f"   gpu_usage_percent存在: {has_gpu_usage}")
                print(f"   gpu_model存在: {has_gpu_model}")
                
                if has_gpu_usage and has_gpu_model:
                    print(f"   gpu_usage_percent值: {system_data['gpu_usage_percent']}")
                    print(f"   gpu_model值: {system_data['gpu_model']}")
                    print("\n🎉 修复成功！GPU字段已恢复")
                else:
                    print("❌ GPU字段仍然缺失")
            else:
                print("❌ 响应中没有system字段")
        else:
            print(f"❌ API响应错误，状态码: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 验证失败: {e}")

if __name__ == "__main__":
    test_gpu_import()
    restart_flask_app()
    verify_fix()