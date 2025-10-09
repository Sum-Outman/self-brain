#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
安装Self Brain AGI系统所需的所有依赖项
"""

import os
import subprocess
import sys
import time

# 安装依赖项的函数
def install_dependencies():
    """安装requirements.txt中的所有依赖项"""
    print("开始安装Self Brain AGI系统的所有依赖项...")
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_file = os.path.join(current_dir, 'requirements.txt')
    
    # 检查requirements.txt文件是否存在
    if not os.path.exists(requirements_file):
        print(f"错误: 找不到{requirements_file}文件")
        return False
    
    try:
        # 使用pip安装依赖项
        print(f"正在从{requirements_file}安装依赖项...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        
        # 分批安装依赖项以提高成功率
        # 首先安装基础依赖
        basic_deps = [
            'flask==2.3.2',
            'flask-socketio==5.3.3',
            'flask-cors==4.0.0',
            'uvicorn==0.22.0',
            'fastapi==0.95.2',
            'starlette==0.27.0',
            'numpy==1.24.3',
            'pandas==2.0.3',
            'matplotlib==3.7.2',
            'psutil==5.9.5',
            'yaml==0.2.5',
            'requests==2.31.0',
            'pyserial==3.5'
        ]
        print("正在安装基础依赖项...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + basic_deps)
        time.sleep(2)
        
        # 安装机器学习依赖
        ml_deps = [
            'pytorch==2.0.1',
            'sentence-transformers==2.2.2',
            'transformers==4.30.2',
            'tensorflow==2.12.0',
            'scikit-learn==1.3.0',
            'torchvision==0.15.2',
            'torchaudio==2.0.2',
            'opencv-python==4.8.0.74',
            'pillow==10.0.0'
        ]
        print("正在安装机器学习依赖项...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + ml_deps)
        time.sleep(2)
        
        # 安装数据库和其他依赖
        db_deps = [
            'neo4j==5.11.0',
            'sqlalchemy==2.0.19',
            'pysqlite3==0.5.0',
            'faiss-cpu==1.7.4',
            'redis==5.0.1'
        ]
        print("正在安装数据库依赖项...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + db_deps)
        time.sleep(2)
        
        # 安装音频和视频处理依赖
        media_deps = [
            'soundfile==0.12.1',
            'librosa==0.10.1',
            'pydub==0.25.1',
            'pyaudio==0.2.13',
            'moviepy==1.0.3'
        ]
        print("正在安装音频和视频处理依赖项...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + media_deps)
        time.sleep(2)
        
        # 安装剩余依赖
        print("正在安装剩余依赖项...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
        
        print("所有依赖项安装成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"安装依赖项时出错: {e}")
        return False
    except Exception as e:
        print(f"发生未知错误: {e}")
        return False

if __name__ == '__main__':
    success = install_dependencies()
    if success:
        print("\n依赖项安装完成，可以通过运行 start_system.py 来启动Self Brain AGI系统")
    else:
        print("\n依赖项安装失败，请检查错误信息并手动安装缺少的依赖项")