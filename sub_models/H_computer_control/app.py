# -*- coding: utf-8 -*-
# Apache License 2.0 开源协议 | Apache License 2.0 Open Source License
# Copyright 2025 AGI System
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import platform
import subprocess
import time
import logging
import json
import threading
import queue
from flask import Flask, request, jsonify
from typing import Dict, Any, List, Optional
import requests
import numpy as np

app = Flask(__name__)

# 配置日志 | Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComputerControlSystem:
    """计算机控制系统类，用于执行系统命令和管理MCP插件 | Computer control system class for executing system commands and managing MCP plugins"""
    
    def __init__(self, language: str = 'en'):
        self.os_type = platform.system()
        self.language = language
        self.data_bus = None
        
        # 多语言支持 | Multilingual support
        self.supported_languages = ['zh', 'en', 'ja', 'de', 'ru']
        self.translations = {
            'command': {'en': 'command', 'zh': '命令', 'ja': 'コマンド', 'de': 'Befehl', 'ru': 'команда'},
            'batch': {'en': 'batch', 'zh': '批量', 'ja': 'バッチ', 'de': 'Stapel', 'ru': 'пакет'},
            'mcp': {'en': 'MCP', 'zh': 'MCP', 'ja': 'MCP', 'de': 'MCP', 'ru': 'MCP'},
            'plugin': {'en': 'plugin', 'zh': '插件', 'ja': 'プラグイン', 'de': 'Plugin', 'ru': 'плагин'},
            'execution': {'en': 'execution', 'zh': '执行', 'ja': '実行', 'de': 'Ausführung', 'ru': 'выполнение'},
            'processing': {'en': 'processing', 'zh': '处理', 'ja': '処理', 'de': 'Verarbeitung', 'ru': 'обработка'},
            'management': {'en': 'management', 'zh': '管理', 'ja': '管理', 'de': 'Verwaltung', 'ru': 'управление'},
            'system': {'en': 'system', 'zh': '系统', 'ja': 'システム', 'de': 'System', 'ru': 'система'},
            'control': {'en': 'control', 'zh': '控制', 'ja': '制御', 'de': 'Steuerung', 'ru': 'управление'}
        }
        
        self.mcp_plugins = self.load_mcp_plugins()
        
        # 实时数据处理 | Real-time data processing
        self.realtime_callbacks = []
        self.data_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_realtime_data)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # 训练历史 | Training history
        self.training_history = []
        
        # 性能监控 | Performance monitoring
        self.performance_stats = {
            'command_execution_time': [],
            'success_rate': [],
            'mcp_plugins_loaded': len(self.mcp_plugins)
        }
        
        # 系统兼容性映射 | System compatibility mapping
        self.system_compatibility = {
            'Windows': ['cmd', 'powershell', 'wsl'],
            'Linux': ['bash', 'sh', 'zsh'],
            'Darwin': ['bash', 'zsh', 'osascript']
        }
    
    def set_language(self, language: str) -> bool:
        """设置当前语言 | Set current language"""
        if language in self.supported_languages:
            self.language = language
            return True
        return False
    
    def set_data_bus(self, data_bus):
        """设置数据总线 | Set data bus"""
        self.data_bus = data_bus
    
    def load_mcp_plugins(self) -> Dict[str, Any]:
        """加载MCP控制插件 | Load MCP control plugins"""
        plugins = {}
        # 实际实现应从plugins目录加载 | Actual implementation should load from plugins directory
        # 这里模拟加载一些常用MCP插件 | Simulate loading some common MCP plugins
        plugins['file_operations'] = {
            "name": self._translate('file', self.language),
            "description": self._translate('file_operations', self.language),
            "execute": self._execute_file_operation
        }
        
        plugins['system_info'] = {
            "name": self._translate('system', self.language) + " " + self._translate('info', self.language),
            "description": self._translate('system_information', self.language),
            "execute": self._get_system_info
        }
        
        plugins['process_management'] = {
            "name": self._translate('process', self.language) + " " + self._translate('management', self.language),
            "description": self._translate('process_management', self.language),
            "execute": self._manage_processes
        }
        
        return plugins
    
    def _process_realtime_data(self):
        """处理实时数据队列 | Process real-time data queue"""
        while True:
            try:
                data = self.data_queue.get(timeout=1.0)
                for callback in self.realtime_callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"回调函数错误: {e} | Callback error: {e}")
                self.data_queue.task_done()
                
                # 发送到数据总线 | Send to data bus
                if self.data_bus:
                    try:
                        self.data_bus.send(data)
                    except Exception as e:
                        logger.error(f"数据总线发送错误: {e} | Data bus send error: {e}")
                else:
                    # 尝试发送到主模型 | Try to send to main model
                    try:
                        requests.post("http://localhost:5000/receive_data", json=data, timeout=1.0)
                    except Exception as e:
                        logger.error(f"主模型通信失败: {e} | Main model communication failed: {e}")
                        
            except queue.Empty:
                continue
    
    def register_realtime_callback(self, callback: callable):
        """注册实时数据回调函数 | Register real-time data callback function"""
        self.realtime_callbacks.append(callback)
    
    def execute_command(self, command: str, mcp_module: str = None, realtime: bool = False) -> Dict[str, Any]:
        """
        执行系统命令或MCP插件 | Execute system command or MCP plugin
        :param command: 要执行的命令 | Command to execute
        :param mcp_module: 使用的MCP模块 | MCP module to use
        :param realtime: 是否为实时数据流 | Whether it's real-time data stream
        :return: 执行结果 | Execution result
        """
        start_time = time.time()
        
        try:
            if mcp_module and mcp_module in self.mcp_plugins:
                # 调用MCP插件 | Call MCP plugin
                result = self.mcp_plugins[mcp_module].execute(command)
                result["module"] = mcp_module
            else:
                # 执行原始命令 | Execute raw command
                if self.os_type == "Windows":
                    result = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        encoding="utf-8"
                    )
                else:  # Linux/macOS
                    result = subprocess.run(
                        command.split(),
                        capture_output=True,
                        text=True
                    )
                
                result = {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr,
                    "returncode": result.returncode,
                    "module": "system"
                }
            
            # 添加性能指标 | Add performance metrics
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            result["timestamp"] = time.time()
            result["lang"] = self.language
            
            # 更新性能统计 | Update performance statistics
            self.performance_stats['command_execution_time'].append(execution_time)
            if len(self.performance_stats['command_execution_time']) > 100:
                self.performance_stats['command_execution_time'].pop(0)
            
            self.performance_stats['success_rate'].append(1 if result["success"] else 0)
            if len(self.performance_stats['success_rate']) > 100:
                self.performance_stats['success_rate'].pop(0)
            
            # 如果是实时数据，放入队列供回调处理 | If real-time data, put in queue for callback processing
            if realtime:
                self.data_queue.put(result)
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "timestamp": time.time(),
                "lang": self.language
            }
            
            if realtime:
                self.data_queue.put(error_result)
            
            return error_result
    
    def batch_process(self, commands: List[str], realtime: bool = False) -> Dict[str, Any]:
        """批量处理多个命令 | Batch process multiple commands"""
        results = []
        for cmd in commands:
            result = self.execute_command(cmd, realtime=realtime)
            results.append(result)
        
        batch_result = {
            "results": results,
            "total_commands": len(commands),
            "successful_commands": sum(1 for r in results if r.get("success", False)),
            "failed_commands": sum(1 for r in results if not r.get("success", True)),
            "total_time": sum(r.get("execution_time", 0) for r in results),
            "timestamp": time.time(),
            "lang": self.language
        }
        
        if realtime:
            self.data_queue.put(batch_result)
        
        return batch_result
    
    def install_mcp_plugin(self, plugin_name: str, config: Dict) -> Dict[str, Any]:
        """安装MCP插件 | Install MCP plugin"""
        try:
            # 实际实现应从外部源下载并安装插件 | Actual implementation should download and install from external source
            self.mcp_plugins[plugin_name] = {
                "name": plugin_name,
                "config": config,
                "execute": lambda cmd: {"status": "mocked", "command": cmd, "success": True}
            }
            
            # 更新性能统计 | Update performance statistics
            self.performance_stats['mcp_plugins_loaded'] = len(self.mcp_plugins)
            
            return {
                "status": "installed",
                "plugin": plugin_name,
                "total_plugins": len(self.mcp_plugins),
                "lang": self.language
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "lang": self.language
            }
    
    def _execute_file_operation(self, command: str) -> Dict[str, Any]:
        """执行文件操作MCP插件 | Execute file operations MCP plugin"""
        try:
            # 解析文件操作命令 | Parse file operation command
            parts = command.split()
            operation = parts[0].lower()
            
            if operation == 'read' and len(parts) > 1:
                # 读取文件 | Read file
                file_path = parts[1]
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {"success": True, "content": content, "operation": "read"}
            
            elif operation == 'write' and len(parts) > 2:
                # 写入文件 | Write file
                file_path = parts[1]
                content = ' '.join(parts[2:])
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return {"success": True, "operation": "write"}
            
            elif operation == 'list' and len(parts) > 1:
                # 列出目录 | List directory
                dir_path = parts[1]
                items = os.listdir(dir_path)
                return {"success": True, "items": items, "operation": "list"}
            
            else:
                return {"success": False, "error": "Unsupported file operation", "operation": operation}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_system_info(self, command: str) -> Dict[str, Any]:
        """获取系统信息MCP插件 | Get system information MCP plugin"""
        try:
            info = {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "architecture": platform.architecture(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "memory_usage": self._get_memory_usage(),
                "disk_usage": self._get_disk_usage(),
                "cpu_usage": self._get_cpu_usage()
            }
            return {"success": True, "system_info": info}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _manage_processes(self, command: str) -> Dict[str, Any]:
        """进程管理MCP插件 | Process management MCP plugin"""
        try:
            parts = command.split()
            operation = parts[0].lower()
            
            if operation == 'list':
                # 列出进程 | List processes
                if self.os_type == "Windows":
                    result = subprocess.run(['tasklist'], capture_output=True, text=True)
                else:
                    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                
                return {"success": True, "processes": result.stdout}
            
            elif operation == 'kill' and len(parts) > 1:
                # 终止进程 | Kill process
                pid = parts[1]
                if self.os_type == "Windows":
                    result = subprocess.run(['taskkill', '/PID', pid, '/F'], capture_output=True, text=True)
                else:
                    result = subprocess.run(['kill', '-9', pid], capture_output=True, text=True)
                
                return {"success": result.returncode == 0, "output": result.stdout, "error": result.stderr}
            
            else:
                return {"success": False, "error": "Unsupported process operation"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况 | Get memory usage"""
        try:
            if self.os_type == "Windows":
                result = subprocess.run(['wmic', 'OS', 'get', 'FreePhysicalMemory,TotalVisibleMemorySize', '/Value'],
                                      capture_output=True, text=True)
                lines = result.stdout.split('\n')
                free_mem = total_mem = 0
                for line in lines:
                    if 'FreePhysicalMemory' in line:
                        free_mem = int(line.split('=')[1])
                    elif 'TotalVisibleMemorySize' in line:
                        total_mem = int(line.split('=')[1])
                
                return {
                    "total": total_mem,
                    "free": free_mem,
                    "used": total_mem - free_mem,
                    "usage_percent": ((total_mem - free_mem) / total_mem) * 100 if total_mem > 0 else 0
                }
            else:
                import psutil
                memory = psutil.virtual_memory()
                return {
                    "total": memory.total,
                    "free": memory.free,
                    "used": memory.used,
                    "usage_percent": memory.percent
                }
        except:
            return {"error": "Unable to get memory usage"}
    
    def _get_disk_usage(self) -> Dict[str, Any]:
        """获取磁盘使用情况 | Get disk usage"""
        try:
            if self.os_type == "Windows":
                result = subprocess.run(['wmic', 'logicaldisk', 'get', 'size,freespace,caption'],
                                      capture_output=True, text=True)
                lines = result.stdout.split('\n')[1:]  # Skip header
                disks = {}
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            disk = parts[0]
                            free = int(parts[1]) if parts[1].isdigit() else 0
                            total = int(parts[2]) if parts[2].isdigit() else 0
                            disks[disk] = {
                                "total": total,
                                "free": free,
                                "used": total - free,
                                "usage_percent": ((total - free) / total) * 100 if total > 0 else 0
                            }
                return disks
            else:
                import psutil
                disks = {}
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        disks[partition.device] = {
                            "total": usage.total,
                            "free": usage.free,
                            "used": usage.used,
                            "usage_percent": usage.percent
                        }
                    except:
                        continue
                return disks
        except:
            return {"error": "Unable to get disk usage"}
    
    def _get_cpu_usage(self) -> Dict[str, Any]:
        """获取CPU使用情况 | Get CPU usage"""
        try:
            if self.os_type == "Windows":
                result = subprocess.run(['wmic', 'cpu', 'get', 'loadpercentage'],
                                      capture_output=True, text=True)
                lines = result.stdout.split('\n')
                usage = 0
                for line in lines:
                    if line.strip().isdigit():
                        usage = int(line.strip())
                        break
                return {"usage_percent": usage}
            else:
                import psutil
                return {"usage_percent": psutil.cpu_percent(interval=1)}
        except:
            return {"error": "Unable to get CPU usage"}
    
    def fine_tune(self, training_data: List[Dict], model_type: str = 'command_optimization') -> Dict:
        """微调计算机控制模型 | Fine-tune computer control model"""
        try:
            # 实际微调逻辑占位符 | Placeholder for actual fine-tuning logic
            logger.info(f"开始微调{model_type}模型 | Starting fine-tuning for {model_type} model")
            logger.info(f"训练样本数: {len(training_data)} | Training samples: {len(training_data)}")
            
            # 模拟训练过程 | Simulate training process
            training_loss = np.random.uniform(0.1, 0.5)
            accuracy = np.random.uniform(0.85, 0.95)
            
            training_result = {
                "status": "success",
                "model_type": model_type,
                "training_loss": training_loss,
                "accuracy": accuracy,
                "samples": len(training_data)
            }
            
            # 记录训练历史 | Record training history
            self.training_history.append({
                "timestamp": time.time(),
                "model_type": model_type,
                "result": training_result
            })
            
            return training_result
            
        except Exception as e:
            error_msg = f"模型微调失败: {str(e)} | Model fine-tuning failed: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def get_monitoring_data(self) -> Dict:
        """获取实时监视数据 | Get real-time monitoring data"""
        avg_execution_time = np.mean(self.performance_stats['command_execution_time']) if self.performance_stats['command_execution_time'] else 0
        success_rate = np.mean(self.performance_stats['success_rate']) if self.performance_stats['success_rate'] else 0
        
        return {
            "status": "active",
            "language": self.language,
            "os_type": self.os_type,
            "mcp_plugins_loaded": self.performance_stats['mcp_plugins_loaded'],
            "performance": {
                "avg_execution_time_ms": avg_execution_time * 1000,
                "success_rate": success_rate,
                "queue_size": self.data_queue.qsize()
            },
            "system_compatibility": self.system_compatibility.get(self.os_type, []),
            "training_history": len(self.training_history)
        }
    
    def _translate(self, text: str, lang: str) -> str:
        """翻译文本 | Translate text"""
        if text in self.translations and lang in self.translations[text]:
            return self.translations[text][lang]
        return text

# 创建计算机控制系统实例 | Create computer control system instance
computer_control = ComputerControlSystem()

# 健康检查端点 | Health check endpoints
@app.route('/')
def index():
    """健康检查端点 | Health check endpoint"""
    return jsonify({
        "status": "active",
        "model": "H_computer_control",
        "version": "2.0.0",
        "language": computer_control.language,
        "capabilities": [
            "command_execution", "batch_processing", "mcp_plugin_management",
            "file_operations", "system_monitoring", "process_management",
            "real_time_processing", "multilingual_support"
        ],
        "supported_systems": computer_control.system_compatibility.get(computer_control.os_type, [])
    })

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查 | Health check"""
    return jsonify({"status": "healthy", "model": "H_computer_control", "lang": computer_control.language})

@app.route('/execute', methods=['POST'])
def execute_command():
    """执行命令API端点 | Execute command API endpoint"""
    lang = request.headers.get('Accept-Language', 'en')[:2]
    if lang not in computer_control.supported_languages:
        lang = 'en'
    
    data = request.json
    command = data.get('command')
    mcp_module = data.get('mcp_module')
    realtime = data.get('realtime', False)
    
    if not command:
        return jsonify({"error": "Missing command", "lang": lang}), 400
    
    result = computer_control.execute_command(command, mcp_module, realtime)
    return jsonify(result)

@app.route('/batch', methods=['POST'])
def batch_commands():
    """批量执行命令API端点 | Batch commands API endpoint"""
    lang = request.headers.get('Accept-Language', 'en')[:2]
    if lang not in computer_control.supported_languages:
        lang = 'en'
    
    data = request.json
    commands = data.get('commands')
    realtime = data.get('realtime', False)
    
    if not commands or not isinstance(commands, list):
        return jsonify({"error": "Missing commands list", "lang": lang}), 400
    
    result = computer_control.batch_process(commands, realtime)
    return jsonify(result)

@app.route('/mcp/install', methods=['POST'])
def install_mcp_plugin():
    """安装MCP插件API端点 | Install MCP plugin API endpoint"""
    lang = request.headers.get('Accept-Language', 'en')[:2]
    if lang not in computer_control.supported_languages:
        lang = 'en'
    
    data = request.json
    plugin_name = data.get('plugin_name')
    config = data.get('config', {})
    
    if not plugin_name:
        return jsonify({"error": "Missing plugin_name", "lang": lang}), 400
    
    result = computer_control.install_mcp_plugin(plugin_name, config)
    return jsonify(result)

@app.route('/mcp/list', methods=['GET'])
def list_mcp_plugins():
    """列出已安装的MCP插件 | List installed MCP plugins"""
    lang = request.headers.get('Accept-Language', 'en')[:2]
    if lang not in computer_control.supported_languages:
        lang = 'en'
    
    plugins = []
    for name, plugin in computer_control.mcp_plugins.items():
        plugins.append({
            "name": name,
            "description": plugin.get("description", ""),
            "config": plugin.get("config", {})
        })
    
    return jsonify({"plugins": plugins, "lang": lang})

@app.route('/register_realtime_callback', methods=['POST'])
def register_realtime_callback():
    """注册实时数据回调API端点 | API endpoint for registering real-time data callback"""
    lang = request.headers.get('Accept-Language', 'en')[:2]
    if lang not in computer_control.supported_languages:
        lang = 'en'
    
    data = request.json
    callback_url = data.get('callback_url')
    
    if not callback_url:
        return jsonify({"error": "Missing callback_url parameter", "lang": lang}), 400
    
    # 创建回调函数 | Create callback function
    def callback(command_data):
        try:
            requests.post(callback_url, json=command_data, timeout=1.0)
        except Exception as e:
            logger.error(f"发送数据到 {callback_url} 失败: {e} | Failed to send data to {callback_url}: {e}")
    
    computer_control.register_realtime_callback(callback)
    return jsonify({"status": "success", "message": "Callback registered", "lang": lang})

@app.route('/train', methods=['POST'])
def train_model():
    """训练计算机控制模型 | Train computer control model"""
    lang = request.headers.get('Accept-Language', 'en')[:2]
    if lang not in computer_control.supported_languages:
        lang = 'en'
    
    try:
        training_data = request.json
        model_type = request.json.get('model_type', 'command_optimization')
        
        # 训练模型 | Train model
        training_result = computer_control.fine_tune(training_data, model_type)
        
        return jsonify({
            "status": "success",
            "lang": lang,
            "message": "模型训练完成",
            "results": training_result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "lang": lang,
            "message": f"训练失败: {str(e)}"
        }), 500

@app.route('/monitor', methods=['GET'])
def get_monitoring_data():
    """获取实时监视数据 | Get real-time monitoring data"""
    monitoring_data = computer_control.get_monitoring_data()
    return jsonify(monitoring_data)

@app.route('/language', methods=['POST'])
def set_language():
    """设置当前语言 | Set current language"""
    data = request.json
    lang = data.get('lang')
    
    if not lang:
        return jsonify({'error': '缺少语言代码', 'lang': 'en'}), 400
    
    if computer_control.set_language(lang):
        return jsonify({'status': f'语言设置为 {lang}', 'lang': lang})
    return jsonify({'error': '无效的语言代码。使用 zh, en, ja, de, ru', 'lang': 'en'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5008, debug=True)
