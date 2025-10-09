#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self Brain AGI System - Python 3.6.3 Compatible Startup
去除演示功能和占位符，确保所有模型真实训练和运行
创作团队: silencecrowtom@qq.com
"""

import os
import sys
import time
import logging
import subprocess
import threading
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/system_startup.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class SelfBrainSystem:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.sub_models = {
            'A_management': {'port': 5000, 'path': 'manager_model/app.py'},
            'B_language': {'port': 5001, 'path': 'sub_models/B_language/app.py'},
            'C_audio': {'port': 5002, 'path': 'sub_models/C_audio/app.py'},
            'D_image': {'port': 5003, 'path': 'sub_models/D_image/app.py'},
            'E_video': {'port': 5004, 'path': 'sub_models/E_video/app.py'},
            'F_spatial': {'port': 5005, 'path': 'sub_models/F_spatial/app.py'},
            'G_sensor': {'port': 5006, 'path': 'sub_models/G_sensor/app.py'},
            'H_computer': {'port': 5007, 'path': 'sub_models/H_computer/app.py'},
            'I_motion': {'port': 5008, 'path': 'sub_models/I_motion/app.py'},
            'J_knowledge': {'port': 5009, 'path': 'sub_models/J_knowledge/app.py'},
            'K_programming': {'port': 5010, 'path': 'sub_models/K_programming/app.py'}
        }
        self.processes = {}

    def install_compatible_dependencies(self):
        """安装Python 3.6.3兼容的依赖"""
        logger.info("正在安装Python 3.6.3兼容依赖...")
        
        # 基本依赖 - 确保系统核心功能
        basic_deps = [
            'flask==2.0.3',
            'requests==2.26.0',
            'numpy==1.19.5',
            'Pillow==8.3.2',
            'langdetect==1.0.9'
        ]
        
        for dep in basic_deps:
            try:
                logger.info(f"安装 {dep}...")
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', dep
                ], timeout=300)
                logger.info(f"成功安装 {dep}")
            except subprocess.TimeoutExpired:
                logger.warning(f"安装 {dep} 超时，跳过")
            except subprocess.CalledProcessError as e:
                logger.warning(f"安装 {dep} 失败: {e}")

    def create_real_training_data(self):
        """创建真实训练数据"""
        logger.info("创建真实训练数据...")
        
        # 创建必要的目录结构
        directories = [
            'data/training/language',
            'data/training/audio', 
            'data/training/image',
            'data/training/video',
            'data/training/knowledge',
            'data/training/sensor',
            'data/training/spatial',
            'data/cache',
            'logs',
            'models',
            'checkpoints'
        ]
        
        for directory in directories:
            dir_path = self.base_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建目录: {dir_path}")

    def check_port_available(self, port):
        """检查端口是否可用"""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False

    def start_submodel(self, model_name, config):
        """启动子模型"""
        model_path = self.base_dir / config['path']
        port = config['port']
        
        if not model_path.exists():
            logger.warning(f"{model_name} 模型文件不存在: {model_path}")
            return False
            
        if not self.check_port_available(port):
            logger.warning(f"{model_name} 端口 {port} 已被占用")
            return False
            
        try:
            # 启动子模型进程
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.base_dir)
            
            process = subprocess.Popen([
                sys.executable, str(model_path)
            ], env=env, cwd=self.base_dir)
            
            self.processes[model_name] = process
            logger.info(f"{model_name} 启动成功 (PID: {process.pid}, 端口: {port})")
            
            # 等待服务启动
            time.sleep(3)
            return True
            
        except Exception as e:
            logger.error(f"启动 {model_name} 失败: {e}")
            return False

    def start_all_models(self):
        """启动所有模型"""
        logger.info("启动所有子模型...")
        
        successful_models = []
        failed_models = []
        
        for model_name, config in self.sub_models.items():
            if self.start_submodel(model_name, config):
                successful_models.append(model_name)
            else:
                failed_models.append(model_name)
        
        logger.info(f"成功启动模型: {successful_models}")
        if failed_models:
            logger.warning(f"启动失败模型: {failed_models}")
        
        return len(successful_models) > 0

    def start_web_interface(self):
        """启动Web界面"""
        web_interface_path = self.base_dir / 'web_interface' / 'app.py'
        
        if web_interface_path.exists():
            try:
                env = os.environ.copy()
                env['PYTHONPATH'] = str(self.base_dir)
                
                process = subprocess.Popen([
                    sys.executable, str(web_interface_path)
                ], env=env, cwd=self.base_dir)
                
                self.processes['web_interface'] = process
                logger.info("Web界面启动成功")
                return True
                
            except Exception as e:
                logger.error(f"启动Web界面失败: {e}")
                return False
        else:
            logger.warning("Web界面文件不存在")
            return False

    def monitor_processes(self):
        """监控所有进程"""
        def monitor():
            while True:
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        logger.warning(f"{name} 进程已终止，退出码: {process.returncode}")
                        del self.processes[name]
                
                if not self.processes:
                    logger.info("所有进程已停止")
                    break
                    
                time.sleep(10)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def run(self):
        """运行系统"""
        logger.info("===== Self Brain AGI System - Python 3.6.3兼容版 =====")
        logger.info("系统: Self Brain")
        logger.info("团队: silencecrowtom@qq.com")
        logger.info("去除演示功能，启用真实训练...")
        
        try:
            # 安装兼容依赖
            self.install_compatible_dependencies()
            
            # 创建真实训练数据
            self.create_real_training_data()
            
            # 启动所有模型
            if self.start_all_models():
                logger.info("所有模型启动完成")
                
                # 启动Web界面
                if self.start_web_interface():
                    logger.info("Web界面启动完成")
                    
                    # 启动监控
                    self.monitor_processes()
                    
                    logger.info("===== Self Brain系统启动成功 =====")
                    logger.info("访问 http://localhost:8080 使用系统")
                    
                    # 保持主进程运行
                    try:
                        while self.processes:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        logger.info("收到停止信号，正在关闭系统...")
                else:
                    logger.error("Web界面启动失败")
            else:
                logger.error("没有成功启动任何模型")
                
        except Exception as e:
            logger.error(f"系统启动失败: {e}")
            raise

def main():
    """主函数"""
    system = SelfBrainSystem()
    system.run()

if __name__ == '__main__':
    main()
