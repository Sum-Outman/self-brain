# Copyright 2025 AGI System Team
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

import os
import shutil
import subprocess
import sys

def rebuild_virtual_env(env_path='myenv'):
    # 删除现有虚拟环境
    if os.path.exists(env_path):
        try:
            shutil.rmtree(env_path)
            print(f"成功删除虚拟环境: {env_path}")
        except Exception as e:
            print(f"删除虚拟环境失败: {e}")
            return False
    
    # 创建新虚拟环境
    try:
        subprocess.check_call([sys.executable, '-m', 'venv', env_path])
        print(f"成功创建虚拟环境: {env_path}")
        return True
    except Exception as e:
        print(f"创建虚拟环境失败: {e}")
        return False

def install_requirements(env_path='myenv'):
    # 获取虚拟环境的pip路径
    pip_path = os.path.join(env_path, 'Scripts', 'pip.exe')
    if not os.path.exists(pip_path):
        print(f"未找到pip: {pip_path}")
        return False
    
    # 安装依赖
    try:
        subprocess.check_call([pip_path, 'install', '-r', 'requirements.txt'])
        print("成功安装依赖")
        return True
    except Exception as e:
        print(f"安装依赖失败: {e}")
        return False

if __name__ == '__main__':
    print("="*50)
    print("重建虚拟环境 | Rebuilding virtual environment")
    print("="*50)
    
    if rebuild_virtual_env():
        if install_requirements():
            print("="*50)
            print("虚拟环境重建成功! | Virtual environment rebuilt successfully!")
            print("="*50)
            sys.exit(0)
    
    print("="*50)
    print("虚拟环境重建失败! | Virtual environment rebuild failed!")
    print("="*50)
    sys.exit(1)
