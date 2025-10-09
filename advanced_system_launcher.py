# -*- coding: utf-8 -*-
# 高级系统启动器 - AGI大脑系统完整启动器
# Advanced System Launcher - Complete AGI Brain System Launcher
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import sys
import subprocess
import threading
import time
import webbrowser
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system_launcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SystemLauncher")

class AdvancedSystemLauncher:
    """
    高级系统启动器 - 完整AGI大脑系统启动器
    Advanced System Launcher - Complete AGI Brain System Launcher
    集成所有组件：核心系统、训练控制器、网页界面
    (Integrates all components: Core System, Training Controller, Web Interface)
    """
    
    def __init__(self):
        """初始化高级系统启动器 | Initialize Advanced System Launcher"""
        # 系统组件进程
        self.processes = {}
        
        # 系统状态
        self.system_status = {
            "is_running": False,
            "start_time": None,
            "components": {},
            "last_update": datetime.now().isoformat()
        }
        
        # 组件配置
        self.components = {
            "core_system": {
                "name": "AGI核心系统",
                "en_name": "AGI Core System",
                "module": "manager_model.core_system",
                "class": "CoreSystem",
                "port": None,
                "requires": []
            },
            "training_controller": {
                "name": "高级训练控制器",
                "en_name": "Advanced Training Controller",
                "module": "training_manager.advanced_train_control",
                "class": "AdvancedTrainingController",
                "port": None,
                "requires": []
            },
            "web_interface": {
                "name": "高级网页界面",
                "en_name": "Advanced Web Interface",
                "module": "web_interface.advanced_web_interface",
                "class": "AdvancedWebInterface",
                "port": 5000,
                "requires": ["core_system", "training_controller"]
            },
            "sub_models": {
                "name": "子模型服务",
                "en_name": "Sub Model Services",
                "processes": {},
                "ports": {
                    "B_language": 5001,
                    "C_audio": 5002,
                    "D_image": 5003,
                    "E_video": 5004,
                    "F_spatial": 5005,
                    "G_sensor": 5006,
                    "H_computer_control": 5007,
                    "I_knowledge": 5008,
                    "J_motion": 5009,
                    "K_programming": 5010
                },
                "requires": []
            }
        }
        
        logger.info("高级系统启动器初始化完成 | Advanced System Launcher initialized")
    
    def start_component(self, component_name: str) -> bool:
        """
        启动系统组件 | Start system component
        
        参数 Parameters:
        component_name: 组件名称 | Component name
        
        返回 Returns:
        启动是否成功 | Whether startup was successful
        """
        try:
            if component_name == "sub_models":
                return self._start_sub_models()
            
            component = self.components[component_name]
            
            # 检查依赖 | Check dependencies
            for dep in component["requires"]:
                if dep not in self.system_status["components"] or not self.system_status["components"][dep]["running"]:
                    logger.error(f"依赖组件 {dep} 未启动 | Dependency component {dep} not started")
                    return False
            
            logger.info(f"启动 {component['name']} | Starting {component['name']}")
            
            if component_name == "core_system":
                # 启动核心系统
                success = self._start_core_system()
            elif component_name == "training_controller":
                # 启动训练控制器
                success = self._start_training_controller()
            elif component_name == "web_interface":
                # 启动网页界面
                success = self._start_web_interface()
            else:
                logger.error(f"未知的组件: {component_name} | Unknown component: {component_name}")
                return False
            
            # 更新状态
            self.system_status["components"][component_name] = {
                "running": success,
                "start_time": datetime.now().isoformat() if success else None,
                "pid": self.processes[component_name].pid if success and component_name in self.processes else None
            }
            
            return success
            
        except Exception as e:
            logger.error(f"启动组件 {component_name} 失败: {e} | Failed to start component {component_name}: {e}")
            return False
    
    def _start_core_system(self) -> bool:
        """启动核心系统 | Start core system"""
        try:
            # 导入核心系统
            from manager_model.core_system import CoreSystem
            
            # 创建核心系统实例
            core_system = CoreSystem(language='zh')
            
            # 启动核心系统
            core_system.start()
            
            # 存储实例引用
            self.processes["core_system"] = core_system
            
            logger.info("AGI核心系统启动成功 | AGI Core System started successfully")
            return True
            
        except Exception as e:
            logger.error(f"启动核心系统失败: {e} | Failed to start core system: {e}")
            return False
    
    def _start_training_controller(self) -> bool:
        """启动训练控制器 | Start training controller"""
        try:
            # 导入训练控制器
            from training_manager.advanced_train_control import AdvancedTrainingController
            
            # 检查是否已经有训练控制器实例（来自web_interface/app.py）
            # 如果有，则使用现有实例而不是创建新实例
            try:
                from web_interface.app import training_control
                if isinstance(training_control, AdvancedTrainingController):
                    # 使用web_interface中已经创建的实例
                    training_controller = training_control
                    logger.info("使用现有的训练控制器实例 | Using existing training controller instance")
                else:
                    # 创建新的训练控制器实例
                    training_controller = AdvancedTrainingController()
                    logger.info("创建新的训练控制器实例 | Creating new training controller instance")
            except ImportError:
                # web_interface/app.py 可能还没有导入，创建新实例
                training_controller = AdvancedTrainingController()
                logger.info("创建新的训练控制器实例 | Creating new training controller instance")
            
            # 存储实例引用
            self.processes["training_controller"] = training_controller
            
            logger.info("高级训练控制器启动成功 | Advanced Training Controller started successfully")
            return True
            
        except Exception as e:
            logger.error(f"启动训练控制器失败: {e} | Failed to start training controller: {e}")
            return False
    
    def _start_web_interface(self) -> bool:
        """启动网页界面 | Start web interface"""
        try:
            # 导入网页界面
            from web_interface.advanced_web_interface import AdvancedWebInterface
            
            # 获取核心系统和训练控制器实例
            core_system = self.processes.get("core_system")
            training_controller = self.processes.get("training_controller")
            
            # 创建网页界面实例
            web_interface = AdvancedWebInterface(
                agi_core=core_system,
                training_controller=training_controller,
                host='127.0.0.1',
                port=5000
            )
            
            # 连接组件
            if core_system:
                web_interface.connect_agi_core(core_system)
            if training_controller:
                web_interface.connect_training_controller(training_controller)
            
            # 启动网页界面线程
            web_thread = threading.Thread(target=web_interface.start, daemon=True)
            web_thread.start()
            
            # 存储实例引用
            self.processes["web_interface"] = web_interface
            self.processes["web_thread"] = web_thread
            
            logger.info("高级网页界面启动成功 | Advanced Web Interface started successfully")
            return True
            
        except Exception as e:
            logger.error(f"启动网页界面失败: {e} | Failed to start web interface: {e}")
            return False
    
    def _start_sub_models(self) -> bool:
        """启动子模型服务 | Start sub model services"""
        try:
            sub_models = self.components["sub_models"]["ports"]
            processes = {}
            
            for model_name, port in sub_models.items():
                model_dir = f"sub_models/{model_name}"
                if not os.path.exists(model_dir):
                    logger.warning(f"子模型目录不存在: {model_dir} | Sub model directory not found: {model_dir}")
                    continue
                
                app_file = os.path.join(model_dir, "app.py")
                if not os.path.exists(app_file):
                    logger.warning(f"子模型应用文件不存在: {app_file} | Sub model app file not found: {app_file}")
                    continue
                
                # 启动子模型服务
                logger.info(f"启动 {model_name} 服务 (端口: {port}) | Starting {model_name} service (port: {port})")
                
                # 设置环境变量
                env = os.environ.copy()
                env['PORT'] = str(port)
                env['PYTHONHASHSEED'] = '0'
                
                # 启动进程
                process = subprocess.Popen(
                    [sys.executable, app_file],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                processes[model_name] = process
                time.sleep(0.5)  # 避免端口冲突
            
            # 存储进程引用
            self.processes["sub_models"] = processes
            
            # 更新状态
            self.system_status["components"]["sub_models"] = {
                "running": True,
                "start_time": datetime.now().isoformat(),
                "process_count": len(processes)
            }
            
            logger.info(f"子模型服务启动成功: {len(processes)} 个模型 | Sub model services started successfully: {len(processes)} models")
            return True
            
        except Exception as e:
            logger.error(f"启动子模型服务失败: {e} | Failed to start sub model services: {e}")
            return False
    
    def start_all_components(self) -> bool:
        """启动所有系统组件 | Start all system components"""
        try:
            self.system_status["is_running"] = True
            self.system_status["start_time"] = datetime.now().isoformat()
            
            logger.info("开始启动所有系统组件 | Starting all system components")
            
            # 启动顺序
            startup_order = [
                "sub_models",
                "core_system", 
                "training_controller",
                "web_interface"
            ]
            
            # 按顺序启动组件
            for component_name in startup_order:
                if not self.start_component(component_name):
                    logger.error(f"组件 {component_name} 启动失败，系统启动中止 | Component {component_name} failed to start, system startup aborted")
                    return False
                
                # 等待组件初始化
                time.sleep(2)
            
            logger.info("所有系统组件启动完成 | All system components started")
            
            # 打开浏览器
            self._open_browser()
            
            return True
            
        except Exception as e:
            logger.error(f"启动所有组件失败: {e} | Failed to start all components: {e}")
            return False
    
    def _open_browser(self):
        """在浏览器中打开系统界面 | Open system interface in browser"""
        try:
            # 等待网页界面启动
            time.sleep(3)
            
            # 打开浏览器
            url = "http://127.0.0.1:5000"
            logger.info(f"在浏览器中打开系统界面: {url} | Opening system interface in browser: {url}")
            
            webbrowser.open(url)
            
        except Exception as e:
            logger.warning(f"无法在浏览器中打开界面: {e} | Cannot open interface in browser: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态 | Get system status"""
        # 更新组件状态
        for component_name in self.components:
            if component_name in self.processes:
                if component_name == "sub_models":
                    # 检查子模型进程状态
                    running_count = 0
                    processes = self.processes[component_name]
                    for model_name, process in processes.items():
                        if process.poll() is None:  # 进程仍在运行
                            running_count += 1
                    
                    self.system_status["components"][component_name]["running"] = running_count > 0
                    self.system_status["components"][component_name]["process_count"] = running_count
                
                elif hasattr(self.processes[component_name], 'system_status'):
                    # 更新组件特定状态
                    component_status = self.processes[component_name].system_status
                    self.system_status["components"][component_name].update(component_status)
        
        # 更新运行时间
        if self.system_status["start_time"]:
            start_time = datetime.fromisoformat(self.system_status["start_time"].replace('Z', '+00:00'))
            self.system_status["uptime_seconds"] = (datetime.now() - start_time).total_seconds()
        
        self.system_status["last_update"] = datetime.now().isoformat()
        
        return self.system_status
    
    def stop_component(self, component_name: str) -> bool:
        """停止系统组件 | Stop system component"""
        try:
            if component_name not in self.processes:
                logger.warning(f"组件 {component_name} 未运行 | Component {component_name} not running")
                return True
            
            logger.info(f"停止 {component_name} | Stopping {component_name}")
            
            if component_name == "sub_models":
                # 停止所有子模型进程
                processes = self.processes[component_name]
                for model_name, process in processes.items():
                    try:
                        process.terminate()
                        process.wait(timeout=5)
                    except:
                        process.kill()
                logger.info("所有子模型服务已停止 | All sub model services stopped")
            
            elif component_name == "core_system":
                # 停止核心系统
                core_system = self.processes[component_name]
                core_system.shutdown()
                logger.info("AGI核心系统已停止 | AGI Core System stopped")
            
            elif component_name == "training_controller":
                # 停止训练控制器
                training_controller = self.processes[component_name]
                training_controller.shutdown()
                logger.info("训练控制器已停止 | Training Controller stopped")
            
            elif component_name == "web_interface":
                # 停止网页界面
                web_interface = self.processes[component_name]
                web_interface.stop()
                logger.info("网页界面已停止 | Web Interface stopped")
            
            # 更新状态
            if component_name in self.system_status["components"]:
                self.system_status["components"][component_name]["running"] = False
            
            # 移除进程引用
            del self.processes[component_name]
            
            return True
            
        except Exception as e:
            logger.error(f"停止组件 {component_name} 失败: {e} | Failed to stop component {component_name}: {e}")
            return False
    
    def stop_all_components(self):
        """停止所有系统组件 | Stop all system components"""
        try:
            logger.info("正在停止所有系统组件 | Stopping all system components")
            
            # 停止顺序（与启动顺序相反）
            stop_order = [
                "web_interface",
                "training_controller",
                "core_system",
                "sub_models"
            ]
            
            # 按顺序停止组件
            for component_name in stop_order:
                if component_name in self.processes:
                    self.stop_component(component_name)
                    time.sleep(1)
            
            self.system_status["is_running"] = False
            logger.info("所有系统组件已停止 | All system components stopped")
            
        except Exception as e:
            logger.error(f"停止所有组件失败: {e} | Failed to stop all components: {e}")
    
    def restart_component(self, component_name: str) -> bool:
        """重启系统组件 | Restart system component"""
        try:
            # 先停止组件
            if component_name in self.processes:
                self.stop_component(component_name)
                time.sleep(2)
            
            # 再启动组件
            return self.start_component(component_name)
            
        except Exception as e:
            logger.error(f"重启组件 {component_name} 失败: {e} | Failed to restart component {component_name}: {e}")
            return False
    
    def monitor_system(self):
        """监控系统状态 | Monitor system status"""
        try:
            while self.system_status["is_running"]:
                # 获取并显示系统状态
                status = self.get_system_status()
                
                print("\n" + "="*60)
                print("AGI大脑系统状态监控 | AGI Brain System Status Monitoring")
                print("="*60)
                
                print(f"系统运行时间: {status.get('uptime_seconds', 0):.0f} 秒")
                print(f"最后更新: {status.get('last_update', 'N/A')}")
                
                print("\n组件状态 | Component Status:")
                print("-" * 40)
                
                for comp_name, comp_status in status.get("components", {}).items():
                    running = comp_status.get("running", False)
                    status_text = "运行中" if running else "已停止"
                    en_status = "Running" if running else "Stopped"
                    
                    comp_info = self.components.get(comp_name, {})
                    comp_display = f"{comp_info.get('name', comp_name)} ({comp_info.get('en_name', comp_name)})"
                    
                    print(f"{comp_display}: {status_text} / {en_status}")
                
                print("="*60)
                
                # 每10秒更新一次
                time.sleep(10)
                
        except Exception as e:
            logger.error(f"系统监控错误: {e} | System monitoring error: {e}")

# 主程序入口
# Main program entry
if __name__ == "__main__":
    # 创建启动器实例
    launcher = AdvancedSystemLauncher()
    
    try:
        print("="*60)
        print("AGI大脑系统启动器 | AGI Brain System Launcher")
        print("="*60)
        print("Copyright 2025 The AGI Brain System Authors")
        print("Licensed under the Apache License, Version 2.0")
        print("="*60)
        
        # 启动所有组件
        print("正在启动系统组件... | Starting system components...")
        success = launcher.start_all_components()
        
        if success:
            print("系统启动成功! | System started successfully!")
            print("系统界面将在浏览器中打开 | System interface will open in browser")
            print("按 Ctrl+C 停止系统 | Press Ctrl+C to stop the system")
            
            # 启动系统监控
            monitor_thread = threading.Thread(target=launcher.monitor_system, daemon=True)
            monitor_thread.start()
            
            # 保持主线程运行
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n接收到中断信号，正在关闭系统... | Received interrupt signal, shutting down system...")
        
        else:
            print("系统启动失败! | System startup failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"系统启动错误: {e} | System startup error: {e}")
        sys.exit(1)
        
    finally:
        # 确保所有组件都被正确关闭
        launcher.stop_all_components()
        print("系统已完全关闭 | System completely shut down")
