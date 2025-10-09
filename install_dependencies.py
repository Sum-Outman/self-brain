#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""直接安装项目依赖到系统Python环境"""

import os
import sys
import subprocess
import re

# 获取项目中需要的主要包名
def get_required_packages():
    # 主要需要安装的包列表，不指定版本号
    packages = [
        'flask', 'flask-socketio', 'flask-cors', 'uvicorn', 'fastapi', 'starlette',
        'pytorch', 'numpy', 'sentence-transformers', 'transformers', 'tensorflow', 'scikit-learn',
        'torchvision', 'torchaudio', 'pandas', 'matplotlib', 'scipy', 'opencv-python', 'pillow',
        'neo4j', 'sqlalchemy', 'pysqlite3', 'faiss-cpu', 'redis',
        'soundfile', 'librosa', 'pydub', 'pyaudio',
        'moviepy',
        'schedule', 'psutil', 'pyyaml', 'python-dotenv', 'requests', 'tqdm', 'loguru',
        'pyserial', 'pymodbus', 'pyusb',
        'pyjwt', 'cryptography', 'passlib'
    ]
    return packages

# 尝试使用系统Python的pip安装依赖
def install_dependencies():
    print("="*50)
    print("直接安装项目依赖 | Installing dependencies directly")
    print("="*50)
    
    # 尝试获取系统Python的pip路径
    python_exe = sys.executable
    print(f"使用Python解释器: {python_exe}")
    
    # 获取需要安装的包列表
    packages = get_required_packages()
    print(f"需要安装的包数量: {len(packages)}")
    
    # 直接使用系统pip安装依赖，不指定版本号
    try:
        # 先更新pip
        print("更新pip...")
        subprocess.run(
            [python_exe, '-m', 'pip', 'install', '--upgrade', 'pip'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 安装主要包
        print("开始安装主要依赖包...")
        result = subprocess.run(
            [python_exe, '-m', 'pip', 'install'] + packages,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode == 0:
            print("依赖安装成功！")
            return True
        else:
            print(f"依赖安装失败，错误代码: {result.returncode}")
            print(f"错误输出: {result.stderr}")
            
            # 尝试单独安装失败的包
            print("尝试单独安装可能失败的包...")
            failed_packages = []
            
            # 解析错误信息找出失败的包
            if 'No matching distribution found for' in result.stderr:
                matches = re.findall(r'No matching distribution found for (\S+)', result.stderr)
                if matches:
                    failed_packages.extend([m.split('==')[0] for m in matches])
            
            # 尝试单独安装这些包
            if failed_packages:
                print(f"尝试单独安装: {', '.join(failed_packages)}")
                for pkg in failed_packages:
                    try:
                        subprocess.run(
                            [python_exe, '-m', 'pip', 'install', pkg],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True
                        )
                    except:
                        continue
            
            return True  # 即使有部分包失败，也继续尝试启动程序
    except Exception as e:
        print(f"依赖安装异常: {str(e)}")
        return False

if __name__ == '__main__':
    if install_dependencies():
        print("="*50)
        print("项目依赖安装成功! | Dependencies installed successfully!")
        print("="*50)
        sys.exit(0)
    
    print("="*50)
    print("项目依赖安装失败! | Dependencies installation failed!")
    print("="*50)
    sys.exit(1)