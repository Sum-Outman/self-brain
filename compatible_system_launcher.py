#!/usr/bin/env python3
"""
Self Brain AGI System - 兼容Python 3.6.3的启动器
修复所有依赖和启动问题，确保系统能正常运行
"""

import os
import sys
import time
import logging
import subprocess
import threading
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SystemLauncher')

class SystemLauncher:
    def __init__(self):
        self.processes = {}
        self.ports = {
            'web_interface': 5000,
            'A_management': 5001,
            'B_language': 5002,
            'C_audio': 5003,
            'D_image': 5004,
            'E_video': 5005,
            'F_spatial': 5006,
            'G_sensor': 5007,
            'H_computer_control': 5008,
            'I_motion_control': 5009,
            'J_knowledge': 5010,
            'K_programming': 5011
        }
        
    def check_python_version(self):
        """检查Python版本兼容性"""
        version = sys.version_info
        logger.info(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
        
        if version.major == 3 and version.minor >= 6:
            logger.info("Python版本兼容")
            return True
        else:
            logger.error(f"不支持的Python版本，需要3.6+")
            return False
    
    def install_dependencies(self):
        """安装兼容的依赖包"""
        logger.info("安装兼容的依赖包...")
        
        try:
            # 使用更新后的requirements文件，兼容Python 3.6
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements_updated.txt'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("依赖安装成功")
                return True
            else:
                logger.error(f"依赖安装失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("依赖安装超时")
            return False
        except Exception as e:
            logger.error(f"依赖安装异常: {e}")
            return False
    
    def start_web_interface(self):
        """启动Web界面"""
        logger.info("启动Web界面...")
        
        try:
            # 直接启动web_interface/app.py，兼容Python 3.6
            process = subprocess.Popen([
                sys.executable, 'web_interface/app.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
            self.processes['web_interface'] = process
            logger.info(f"Web界面启动成功 (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"启动Web界面失败: {e}")
            return False
    
    def start_model(self, model_name, port):
        """启动单个模型"""
        logger.info(f"启动模型 {model_name} 在端口 {port}...")
        
        try:
            # 设置环境变量
            env = os.environ.copy()
            env['PORT'] = str(port)
            env['HOST'] = '0.0.0.0'
            env['DEBUG'] = 'false'
            
            # 构建模型启动命令
            if model_name == 'A_management':
                script_path = 'manager_model/app.py'
            else:
                # 其他模型使用统一的启动模式
                script_path = f'sub_models/{model_name}/app.py'
            
            # 检查模型文件是否存在
            if not os.path.exists(script_path):
                logger.warning(f"模型文件不存在: {script_path}，跳过启动")
                return False
            
            process = subprocess.Popen([
                sys.executable, script_path
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
            self.processes[model_name] = process
            logger.info(f"模型 {model_name} 启动成功 (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"启动模型 {model_name} 失败: {e}")
            return False
    
    def check_process_health(self, name, process):
        """检查进程健康状态"""
        if process.poll() is None:
            return True  # 进程仍在运行
        else:
            return_code = process.returncode
            logger.warning(f"进程 {name} 已退出，返回代码: {return_code}")
            return False
    
    def monitor_processes(self):
        """监控所有进程状态"""
        logger.info("开始监控进程状态...")
        
        while True:
            time.sleep(10)  # 每10秒检查一次
            
            for name, process in list(self.processes.items()):
                if not self.check_process_health(name, process):
                    # 进程已退出，尝试重启
                    logger.info(f"尝试重启进程: {name}")
                    
                    if name == 'web_interface':
                        self.start_web_interface()
                    else:
                        port = self.ports.get(name, 5000)
                        self.start_model(name, port)
            
            # 记录当前运行状态
            running_count = sum(1 for p in self.processes.values() if p.poll() is None)
            logger.info(f"当前运行进程数: {running_count}/{len(self.processes)}")
    
    def start_all_models(self):
        """启动所有模型"""
        logger.info("启动所有模型...")
        
        # 先启动Web界面
        if not self.start_web_interface():
            logger.error("Web界面启动失败，系统无法继续")
            return False
        
        # 等待Web界面启动
        time.sleep(5)
        
        # 启动所有模型
        success_count = 0
        for model_name, port in self.ports.items():
            if model_name == 'web_interface':
                continue  # 已经启动过了
                
            if self.start_model(model_name, port):
                success_count += 1
            time.sleep(2)  # 间隔启动，避免端口冲突
        
        logger.info(f"成功启动 {success_count}/{len(self.ports)-1} 个模型")
        return success_count > 0
    
    def stop_all_processes(self):
        """停止所有进程"""
        logger.info("停止所有进程...")
        
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"进程 {name} 已停止")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"进程 {name} 被强制终止")
            except Exception as e:
                logger.error(f"停止进程 {name} 时出错: {e}")
        
        self.processes.clear()
    
    def run(self):
        """运行系统启动器"""
        logger.info("=== Self Brain AGI System 启动 ===")
        
        # 检查Python版本
        if not self.check_python_version():
            logger.error("Python版本不兼容，系统无法启动")
            return False
        
        # 安装依赖
        if not self.install_dependencies():
            logger.error("依赖安装失败，系统无法启动")
            return False
        
        # 启动所有模型
        if not self.start_all_models():
            logger.error("模型启动失败")
            return False
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        
        logger.info("=== Self Brain AGI System 启动完成 ===")
        logger.info(f"Web界面: http://localhost:{self.ports['web_interface']}")
        logger.info("系统正在运行，按 Ctrl+C 停止")
        
        try:
            # 主线程等待
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("接收到停止信号")
        finally:
            self.stop_all_processes()
        
        return True

def main():
    """主函数"""
    launcher = SystemLauncher()
    
    try:
        success = launcher.run()
        if success:
            logger.info("系统正常退出")
            return 0
        else:
            logger.error("系统启动失败")
            return 1
    except Exception as e:
        logger.error(f"系统运行异常: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
