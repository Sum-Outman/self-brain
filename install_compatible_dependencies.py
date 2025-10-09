#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
安装Self Brain AGI系统的兼容Python 3.6的依赖项
"""

import os
import subprocess
import sys
import time
import io

# 安装依赖项的函数
def install_dependencies():
    """安装兼容Python 3.6的依赖项"""
    print("开始安装Self Brain AGI系统的兼容Python 3.6的依赖项...")
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_file = os.path.join(current_dir, 'requirements_py36.txt')
    
    # 检查requirements.txt文件是否存在
    if not os.path.exists(requirements_file):
        print(f"错误: 找不到{requirements_file}文件")
        return False
    
    try:
        # 使用pip安装依赖项，设置超时和重试次数
        print("正在升级pip...")
        subprocess.call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip==21.3.1'])
        
        print(f"正在从{requirements_file}安装依赖项...")
        
        # 读取requirements文件并逐个安装包，这样即使某个包安装失败，其他包仍然可以继续安装
        with open(requirements_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 过滤掉注释行和空行
        packages = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                packages.append(line)
        
        # 统计成功和失败的包
        success_count = 0
        failed_packages = []
        
        # 逐个安装包
        for package in packages:
            print(f"\n正在安装: {package}")
            try:
                # 使用Python 3.6兼容的方式调用subprocess
                # 设置stdout和stderr的管道
                process = subprocess.Popen([sys.executable, '-m', 'pip', 'install', 
                                          '--no-cache-dir', '--timeout', '60', package],
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE, 
                                        universal_newlines=True)
                
                # 等待进程完成
                stdout, stderr = process.communicate(timeout=120)  # 增加超时时间
                
                if process.returncode == 0:
                    print(f"✓ 成功安装: {package}")
                    success_count += 1
                else:
                    print(f"✗ 安装失败: {package}")
                    # 只打印前500个字符的错误信息
                    error_msg = stderr[:500] if stderr else "无错误信息"
                    print(f"错误信息: {error_msg}...")
                    failed_packages.append(package)
                
                # 小暂停，避免过于频繁的安装请求
                time.sleep(1)
                
            except Exception as e:
                print(f"✗ 安装失败: {package}")
                print(f"异常信息: {str(e)}")
                failed_packages.append(package)
        
        # 打印安装结果摘要
        print("\n" + "="*60)
        print(f"依赖项安装结果摘要:")
        print(f"总包数: {len(packages)}")
        print(f"成功安装: {success_count}")
        print(f"安装失败: {len(failed_packages)}")
        
        if failed_packages:
            print(f"失败的包列表:")
            for pkg in failed_packages:
                print(f"  - {pkg}")
            print("\n注意：有些包可能需要特定的系统依赖或环境。")
            print("尽管有一些包安装失败，但系统可能仍然可以运行。")
        
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"发生未知错误: {e}")
        return False

if __name__ == '__main__':
    success = install_dependencies()
    if success:
        print("\n依赖项安装完成，可以尝试运行 start_system.py 来启动Self Brain AGI系统")
    else:
        print("\n依赖项安装过程中出现错误，请检查错误信息")