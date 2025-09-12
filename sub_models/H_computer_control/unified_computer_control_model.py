# -*- coding: utf-8 -*-
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0

"""
统一计算机控制模型 | Unified Computer Control Model
整合标准模式和增强模式功能
"""

import subprocess
import psutil
import platform
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import logging
import threading
from dataclasses import dataclass

@dataclass
class SystemInfo:
    """系统信息数据结构"""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    processes: int
    boot_time: float

@dataclass
class ControlCommand:
    """控制命令数据结构"""
    command: str
    parameters: Dict[str, Any]
    target: str
    priority: int = 1
    timeout: int = 30

class UnifiedComputerControlModel:
    """
    统一计算机控制模型
    支持系统监控、进程管理、文件操作、网络控制等功能
    """
    
    def __init__(self, mode: str = "standard", config: Optional[Dict] = None):
        self.mode = mode
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 系统监控
        self.monitoring = False
        self.monitor_thread = None
        self.system_history = []
        self.max_history = 1000
        
        # 进程管理
        self.managed_processes = {}
        self.process_callbacks = {}
        
        # 配置
        self.update_interval = self.config.get("update_interval", 5.0)
        self.warning_threshold = self.config.get("warning_threshold", 80.0)
        
    def get_system_info(self) -> SystemInfo:
        """获取系统信息"""
        try:
            # CPU和内存
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # 磁盘使用
            disk_usage = psutil.disk_usage('/').percent
            
            # 网络IO
            network_io = psutil.net_io_counters()._asdict()
            
            # 进程数
            processes = len(psutil.pids())
            
            # 启动时间
            boot_time = psutil.boot_time()
            
            info = SystemInfo(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage=disk_usage,
                network_io=network_io,
                processes=processes,
                boot_time=boot_time
            )
            
            # 记录历史
            self.system_history.append({
                "timestamp": time.time(),
                "info": info.__dict__
            })
            
            if len(self.system_history) > self.max_history:
                self.system_history.pop(0)
            
            return info
            
        except Exception as e:
            self.logger.error(f"获取系统信息失败: {e}")
            return None
    
    def execute_command(self, command: ControlCommand) -> Dict[str, Any]:
        """执行控制命令"""
        try:
            if command.target == "system":
                return self._execute_system_command(command)
            elif command.target == "file":
                return self._execute_file_command(command)
            elif command.target == "network":
                return self._execute_network_command(command)
            elif command.target == "process":
                return self._execute_process_command(command)
            else:
                return {"error": f"未知目标: {command.target}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def start_system_monitoring(self):
        """启动系统监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("系统监控已启动")
    
    def stop_system_monitoring(self):
        """停止系统监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("系统监控已停止")
    
    def get_process_list(self) -> List[Dict[str, Any]]:
        """获取进程列表"""
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    pinfo = proc.info
                    processes.append({
                        "pid": pinfo['pid'],
                        "name": pinfo['name'],
                        "cpu_percent": pinfo['cpu_percent'],
                        "memory_percent": pinfo['memory_percent']
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            self.logger.error(f"获取进程列表失败: {e}")
        
        return processes
    
    def kill_process(self, pid: int) -> bool:
        """终止进程"""
        try:
            process = psutil.Process(pid)
            process.terminate()
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    def create_directory(self, path: str) -> bool:
        """创建目录"""
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception:
            return False
    
    def list_directory(self, path: str) -> List[Dict[str, Any]]:
        """列出目录内容"""
        try:
            items = []
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                stat = os.stat(item_path)
                items.append({
                    "name": item,
                    "type": "directory" if os.path.isdir(item_path) else "file",
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
            return items
        except Exception as e:
            self.logger.error(f"列出目录失败: {e}")
            return []
    
    def get_network_info(self) -> Dict[str, Any]:
        """获取网络信息"""
        try:
            interfaces = psutil.net_if_addrs()
            stats = psutil.net_if_stats()
            
            network_info = {}
            for interface, addrs in interfaces.items():
                if interface in stats:
                    network_info[interface] = {
                        "addresses": [addr.address for addr in addrs],
                        "is_up": stats[interface].isup,
                        "speed": stats[interface].speed,
                        "mtu": stats[interface].mtu
                    }
            
            return network_info
            
        except Exception as e:
            self.logger.error(f"获取网络信息失败: {e}")
            return {}
    
    def shutdown_system(self) -> bool:
        """关闭系统"""
        try:
            if platform.system() == "Windows":
                subprocess.run(["shutdown", "/s", "/t", "60"], check=True)
            else:
                subprocess.run(["shutdown", "-h", "+1"], check=True)
            return True
        except Exception:
            return False
    
    def restart_system(self) -> bool:
        """重启系统"""
        try:
            if platform.system() == "Windows":
                subprocess.run(["shutdown", "/r", "/t", "60"], check=True)
            else:
                subprocess.run(["reboot"], check=True)
            return True
        except Exception:
            return False
    
    def get_system_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取系统历史记录"""
        return self.system_history[-limit:]
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        info = self.get_system_info()
        if info:
            return {
                "system_info": info.__dict__,
                "monitoring_active": self.monitoring,
                "managed_processes": len(self.managed_processes),
                "history_length": len(self.system_history),
                "mode": self.mode
            }
        return {"error": "无法获取系统信息"}
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                info = self.get_system_info()
                
                # 检查警告阈值
                if info.cpu_percent > self.warning_threshold:
                    self.logger.warning(f"CPU使用率过高: {info.cpu_percent}%")
                
                if info.memory_percent > self.warning_threshold:
                    self.logger.warning(f"内存使用率过高: {info.memory_percent}%")
                
                if info.disk_usage > self.warning_threshold:
                    self.logger.warning(f"磁盘使用率过高: {info.disk_usage}%")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                time.sleep(self.update_interval)
    
    def _execute_system_command(self, command: ControlCommand) -> Dict[str, Any]:
        """执行系统命令"""
        cmd = command.command
        
        if cmd == "get_info":
            return {"result": self.get_system_info().__dict__}
        elif cmd == "shutdown":
            return {"result": self.shutdown_system()}
        elif cmd == "restart":
            return {"result": self.restart_system()}
        else:
            return {"error": f"未知系统命令: {cmd}"}
    
    def _execute_file_command(self, command: ControlCommand) -> Dict[str, Any]:
        """执行文件命令"""
        cmd = command.command
        params = command.parameters
        
        if cmd == "create_dir":
            path = params.get("path", "")
            return {"result": self.create_directory(path)}
        elif cmd == "list_dir":
            path = params.get("path", ".")
            return {"result": self.list_directory(path)}
        else:
            return {"error": f"未知文件命令: {cmd}"}
    
    def _execute_network_command(self, command: ControlCommand) -> Dict[str, Any]:
        """执行网络命令"""
        cmd = command.command
        
        if cmd == "get_info":
            return {"result": self.get_network_info()}
        else:
            return {"error": f"未知网络命令: {cmd}"}
    
    def _execute_process_command(self, command: ControlCommand) -> Dict[str, Any]:
        """执行进程命令"""
        cmd = command.command
        params = command.parameters
        
        if cmd == "list":
            return {"result": self.get_process_list()}
        elif cmd == "kill":
            pid = params.get("pid")
            if pid:
                return {"result": self.kill_process(pid)}
            else:
                return {"error": "缺少PID参数"}
        else:
            return {"error": f"未知进程命令: {cmd}"}