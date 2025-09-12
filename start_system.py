# Copyright 2025 The AI Management System Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 系统启动脚本
# System Startup Script

import os
import subprocess
import sys
import threading
import time
import webbrowser

def start_submodel(model_name, port):
    """启动子模型服务 | Start submodel service"""
    model_dir = f"sub_models/{model_name}"
    if not os.path.exists(model_dir):
        print(f"错误: {model_name} 目录不存在 | Error: {model_name} directory not found")
        return
        
    app_file = os.path.join(model_dir, "app.py")
    if not os.path.exists(app_file):
        print(f"错误: {model_name} app.py 文件不存在 | Error: {model_name} app.py file not found")
        return
        
    print(f"启动 {model_name} 服务 (端口: {port}) | Starting {model_name} service (port: {port})")
    # 使用系统Python并设置随机数种子以避免初始化问题 | Use system Python with random seed to avoid initialization issues
    print(f"警告: 使用系统Python启动{model_name}服务 | Warning: Using system Python to start {model_name} service")
    # 添加随机数种子参数
    env = os.environ.copy()
    env['PORT'] = str(port)
    env['PYTHONHASHSEED'] = '0'
    subprocess.Popen([sys.executable, app_file], env=env)

def start_all_models():
    """启动所有子模型 | Start all submodels"""
    models = {
        "B_language": 5002,
        "C_audio": 5003,
        "D_image": 5004,
        "E_video": 5005,
        "F_spatial": 5006,
        "G_sensor": 5007,
        "H_computer_control": 5008,
        "I_knowledge": 5009,
        "J_motion": 5010,
        "K_programming": 5011
    }
    
    threads = []
    for model, port in models.items():
        thread = threading.Thread(target=start_submodel, args=(model, port))
        thread.start()
        threads.append(thread)
        time.sleep(0.5)  # 避免端口冲突
        
    for thread in threads:
        thread.join()

def main():
    """主启动函数 | Main startup function"""
    print("="*50)
    print("AI管理系统启动中... | AI Management System starting...")
    print("="*50)
    
    # 启动所有子模型
    start_all_models()
    
    # 启动Web界面
    print("启动Web界面 (端口: 5000) | Starting web interface (port: 5000)")
    
    def start_web():
        os.chdir(os.path.join(os.path.dirname(__file__), 'web_interface'))
        subprocess.run([sys.executable, 'app.py'])
    
    web_thread = threading.Thread(target=start_web)
    web_thread.daemon = True
    web_thread.start()
    
    # 打开浏览器
    time.sleep(3)  # 等待服务启动
    print("在浏览器中打开系统界面 | Opening system interface in browser")
    webbrowser.open("http://localhost:5000")
    
    print("="*50)
    print("系统启动完成! | System startup completed!")
    print("="*50)
    
    # 保持主线程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n系统关闭中... | System shutting down...")

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()
