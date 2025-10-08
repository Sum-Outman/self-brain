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

"""
执行器控制模块
提供对各类执行器的统一控制接口
"""

import os
import sys
import time
import threading
import logging
import json
import math
import numpy as np
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Tuple

# 导入设备接口管理器
from H_computer_control.device_interface import device_interface_manager, DeviceProtocol

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ActuatorControl')

class ActuatorType(Enum):
    STEPPER = "stepper"
    SERVO = "servo"
    DC_MOTOR = "dc_motor"
    LINEAR_ACTUATOR = "linear_actuator"
    SOLENOID = "solenoid"
    PNEUMATIC = "pneumatic"
    HYDRAULIC = "hydraulic"
    ROBOTIC_ARM = "robotic_arm"

class ActuatorStatus(Enum):
    IDLE = "idle"
    MOVING = "moving"
    BLOCKED = "blocked"
    ERROR = "error"
    WARNING = "warning"
    INITIALIZING = "initializing"
    CALIBRATING = "calibrating"
    MAINTENANCE = "maintenance"

class ActuatorControlException(Exception):
    """执行器控制异常类"""
    pass

class ActuatorController:
    """
    执行器控制器基类
    提供执行器控制的基本接口
    """
    def __init__(self, actuator_id: str, actuator_type: ActuatorType, config: Dict[str, Any]):
        """
        初始化执行器控制器
        
        参数:
            actuator_id: 执行器ID
            actuator_type: 执行器类型
            config: 配置参数
        """
        self.actuator_id = actuator_id
        self.actuator_type = actuator_type
        self.config = config
        self.status = ActuatorStatus.IDLE
        self.current_position = 0.0  # 当前位置
        self.target_position = 0.0   # 目标位置
        self.velocity = 0.0          # 当前速度
        self.max_velocity = config.get('max_velocity', 100.0)  # 最大速度
        self.acceleration = config.get('acceleration', 50.0)   # 加速度
        self.error_message = None
        
        # 物理限制
        self.min_position = config.get('min_position', -float('inf'))  # 最小位置
        self.max_position = config.get('max_position', float('inf'))   # 最大位置
        self.min_velocity = config.get('min_velocity', 0.0)            # 最小速度
        
        # 传感器配置
        self.sensors = config.get('sensors', {})
        self.sensor_readings = {}
        
        # 运动规划
        self.motion_plan = None
        self.motion_start_time = 0.0
        
        # 控制参数
        self.pid_params = config.get('pid', {'p': 1.0, 'i': 0.0, 'd': 0.0})
        self.pid_integral = 0.0
        self.pid_last_error = 0.0
        
        # 线程控制
        self.lock = threading.Lock()
        self.control_thread = None
        self.stop_event = threading.Event()
        
        # 回调函数
        self.position_changed_callback = None
        self.status_changed_callback = None
        self.motion_completed_callback = None
        self.sensor_data_callback = None
        
        # 设备接口
        self.device_interface = None
        
        logger.info(f"执行器控制器初始化 - ID: {actuator_id}, 类型: {actuator_type.value}")
        
    def initialize(self) -> bool:
        """初始化执行器"""
        with self.lock:
            if self.status != ActuatorStatus.IDLE:
                logger.warning(f"执行器已经初始化 - ID: {self.actuator_id}")
                return True
                
            try:
                self.status = ActuatorStatus.INITIALIZING
                self._notify_status_change()
                
                # 初始化设备接口
                if 'device_config' in self.config:
                    device_config = self.config['device_config']
                    protocol = device_config.get('protocol', 'serial')
                    
                    self.device_interface = device_interface_manager.create_device_interface(
                        f"{self.actuator_id}_device",
                        protocol,
                        device_config.get('params', {})
                    )
                    
                    # 连接设备
                    if not self.device_interface.connect():
                        raise ActuatorControlException(f"设备连接失败: {self.device_interface.error_message}")
                        
                    # 设置数据回调
                    self.device_interface.set_data_received_callback(self._handle_device_data)
                    
                # 执行初始化程序
                self._initialize_actuator()
                
                # 启动控制线程
                self.stop_event.clear()
                self.control_thread = threading.Thread(target=self._control_loop)
                self.control_thread.daemon = True
                self.control_thread.start()
                
                # 校准传感器
                if self.sensors:
                    self._calibrate_sensors()
                    
                self.status = ActuatorStatus.IDLE
                logger.info(f"执行器初始化成功 - ID: {self.actuator_id}")
                self._notify_status_change()
                return True
                
            except Exception as e:
                self.status = ActuatorStatus.ERROR
                self.error_message = str(e)
                
                logger.error(f"执行器初始化失败 - ID: {self.actuator_id}, 错误: {str(e)}")
                self._notify_status_change()
                return False
                
    def shutdown(self) -> bool:
        """关闭执行器"""
        with self.lock:
            try:
                # 停止控制线程
                self.stop_event.set()
                if self.control_thread:
                    self.control_thread.join(timeout=2.0)
                    
                # 断开设备连接
                if self.device_interface:
                    self.device_interface.disconnect()
                    
                self.status = ActuatorStatus.IDLE
                self.error_message = None
                
                logger.info(f"执行器已关闭 - ID: {self.actuator_id}")
                self._notify_status_change()
                return True
                
            except Exception as e:
                logger.error(f"执行器关闭失败 - ID: {self.actuator_id}, 错误: {str(e)}")
                return False
                
    def move_to(self, position: float, velocity: Optional[float] = None) -> bool:
        """
        移动到指定位置
        
        参数:
            position: 目标位置
            velocity: 移动速度（可选）
        
        返回:
            是否成功启动移动
        """
        with self.lock:
            # 检查状态
            if self.status in [ActuatorStatus.ERROR, ActuatorStatus.MAINTENANCE]:
                logger.warning(f"执行器状态不允许移动 - ID: {self.actuator_id}, 状态: {self.status.value}")
                return False
                
            # 检查位置限制
            if position < self.min_position or position > self.max_position:
                logger.warning(f"目标位置超出限制 - ID: {self.actuator_id}, 位置: {position}")
                return False
                
            # 设置目标位置和速度
            self.target_position = position
            if velocity is not None:
                self.max_velocity = max(self.min_velocity, min(velocity, self.config.get('max_velocity', 100.0)))
                
            # 生成运动规划
            self._generate_motion_plan()
            
            # 更新状态
            self.status = ActuatorStatus.MOVING
            self.motion_start_time = time.time()
            
            logger.info(f"执行器开始移动 - ID: {self.actuator_id}, 目标位置: {position}, 速度: {self.max_velocity}")
            self._notify_status_change()
            return True
            
    def move_relative(self, distance: float, velocity: Optional[float] = None) -> bool:
        """
        相对移动指定距离
        
        参数:
            distance: 移动距离
            velocity: 移动速度（可选）
        
        返回:
            是否成功启动移动
        """
        target_position = self.current_position + distance
        return self.move_to(target_position, velocity)
        
    def stop(self) -> bool:
        """停止执行器"""
        with self.lock:
            if self.status != ActuatorStatus.MOVING:
                logger.warning(f"执行器未在移动 - ID: {self.actuator_id}")
                return True
                
            try:
                # 取消运动规划
                self.motion_plan = None
                
                # 发送停止命令到设备
                if self.device_interface:
                    self._send_stop_command()
                    
                # 更新状态
                self.status = ActuatorStatus.IDLE
                self.velocity = 0.0
                
                logger.info(f"执行器已停止 - ID: {self.actuator_id}")
                self._notify_status_change()
                return True
                
            except Exception as e:
                logger.error(f"执行器停止失败 - ID: {self.actuator_id}, 错误: {str(e)}")
                return False
                
    def get_status(self) -> Dict[str, Any]:
        """获取执行器状态"""
        with self.lock:
            return {
                'actuator_id': self.actuator_id,
                'type': self.actuator_type.value,
                'status': self.status.value,
                'current_position': self.current_position,
                'target_position': self.target_position,
                'velocity': self.velocity,
                'error_message': self.error_message,
                'sensor_readings': self.sensor_readings,
                'config': self.config
            }
            
    def read_sensors(self) -> Dict[str, Any]:
        """读取所有传感器数据"""
        with self.lock:
            # 读取传感器数据
            self._read_sensors()
            return self.sensor_readings.copy()
            
    def set_position_changed_callback(self, callback: Callable[[str, float], None]):
        """设置位置变更回调函数"""
        self.position_changed_callback = callback
        
    def set_status_changed_callback(self, callback: Callable[[str, Dict], None]):
        """设置状态变更回调函数"""
        self.status_changed_callback = callback
        
    def set_motion_completed_callback(self, callback: Callable[[str, float], None]):
        """设置运动完成回调函数"""
        self.motion_completed_callback = callback
        
    def set_sensor_data_callback(self, callback: Callable[[str, Dict], None]):
        """设置传感器数据回调函数"""
        self.sensor_data_callback = callback
        
    def _initialize_actuator(self):
        """初始化执行器硬件"""
        # 子类实现具体的初始化逻辑
        pass
        
    def _generate_motion_plan(self):
        """生成运动规划"""
        # 简单的梯形速度规划
        start_position = self.current_position
        end_position = self.target_position
        distance = end_position - start_position
        
        # 计算加减速阶段的距离
        accel_distance = (self.max_velocity ** 2) / (2 * self.acceleration)
        
        # 检查是否需要完全加减速
        if abs(distance) < 2 * accel_distance:
            # 三角形速度规划
            peak_velocity = math.sqrt(self.acceleration * abs(distance))
            self.motion_plan = {
                'type': 'triangle',
                'start_position': start_position,
                'end_position': end_position,
                'distance': distance,
                'peak_velocity': peak_velocity,
                'acceleration': self.acceleration,
                'accel_time': peak_velocity / self.acceleration,
                'total_time': 2 * peak_velocity / self.acceleration
            }
        else:
            # 梯形速度规划
            self.motion_plan = {
                'type': 'trapezoid',
                'start_position': start_position,
                'end_position': end_position,
                'distance': distance,
                'max_velocity': self.max_velocity,
                'acceleration': self.acceleration,
                'accel_time': self.max_velocity / self.acceleration,
                'cruise_time': (abs(distance) - 2 * accel_distance) / self.max_velocity,
                'total_time': 2 * self.max_velocity / self.acceleration + (abs(distance) - 2 * accel_distance) / self.max_velocity
            }
            
    def _calculate_position(self, elapsed_time: float) -> float:
        """根据运动规划计算当前位置"""
        if not self.motion_plan:
            return self.current_position
            
        plan = self.motion_plan
        
        if elapsed_time >= plan['total_time']:
            # 运动完成
            return plan['end_position']
            
        if plan['type'] == 'triangle':
            # 三角形速度规划
            if elapsed_time <= plan['accel_time']:
                # 加速阶段
                position = plan['start_position'] + 0.5 * plan['acceleration'] * elapsed_time ** 2 * (1 if plan['distance'] > 0 else -1)
            else:
                # 减速阶段
                decel_time = elapsed_time - plan['accel_time']
                position = plan['start_position'] + (plan['distance'] / 2 + plan['peak_velocity'] * decel_time - 0.5 * plan['acceleration'] * decel_time ** 2) * (1 if plan['distance'] > 0 else -1)
                
        else:  # trapezoid
            # 梯形速度规划
            if elapsed_time <= plan['accel_time']:
                # 加速阶段
                position = plan['start_position'] + 0.5 * plan['acceleration'] * elapsed_time ** 2 * (1 if plan['distance'] > 0 else -1)
            elif elapsed_time <= plan['accel_time'] + plan['cruise_time']:
                # 匀速阶段
                cruise_time = elapsed_time - plan['accel_time']
                position = plan['start_position'] + (0.5 * plan['acceleration'] * plan['accel_time'] ** 2 + plan['max_velocity'] * cruise_time) * (1 if plan['distance'] > 0 else -1)
            else:
                # 减速阶段
                decel_time = elapsed_time - plan['accel_time'] - plan['cruise_time']
                position = plan['start_position'] + (0.5 * plan['acceleration'] * plan['accel_time'] ** 2 + plan['max_velocity'] * plan['cruise_time'] + plan['max_velocity'] * decel_time - 0.5 * plan['acceleration'] * decel_time ** 2) * (1 if plan['distance'] > 0 else -1)
                
        return position
        
    def _calculate_velocity(self, elapsed_time: float) -> float:
        """根据运动规划计算当前速度"""
        if not self.motion_plan:
            return 0.0
            
        plan = self.motion_plan
        
        if elapsed_time >= plan['total_time']:
            # 运动完成
            return 0.0
            
        if plan['type'] == 'triangle':
            # 三角形速度规划
            if elapsed_time <= plan['accel_time']:
                # 加速阶段
                velocity = plan['acceleration'] * elapsed_time
            else:
                # 减速阶段
                decel_time = elapsed_time - plan['accel_time']
                velocity = plan['peak_velocity'] - plan['acceleration'] * decel_time
                
        else:  # trapezoid
            # 梯形速度规划
            if elapsed_time <= plan['accel_time']:
                # 加速阶段
                velocity = plan['acceleration'] * elapsed_time
            elif elapsed_time <= plan['accel_time'] + plan['cruise_time']:
                # 匀速阶段
                velocity = plan['max_velocity']
            else:
                # 减速阶段
                decel_time = elapsed_time - plan['accel_time'] - plan['cruise_time']
                velocity = plan['max_velocity'] - plan['acceleration'] * decel_time
                
        # 调整速度方向
        if plan['distance'] < 0:
            velocity = -velocity
            
        return velocity
        
    def _control_loop(self):
        """控制循环"""
        last_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                delta_time = current_time - last_time
                last_time = current_time
                
                # 读取传感器数据
                self._read_sensors()
                
                # 检查是否被阻挡
                if self._check_blocked():
                    self.status = ActuatorStatus.BLOCKED
                    self._notify_status_change()
                    
                    # 发送停止命令
                    self.stop()
                    continue
                    
                # 运动控制
                if self.status == ActuatorStatus.MOVING:
                    elapsed_time = current_time - self.motion_start_time
                    
                    # 计算目标位置
                    target_position = self._calculate_position(elapsed_time)
                    target_velocity = self._calculate_velocity(elapsed_time)
                    
                    # 计算位置误差
                    error = target_position - self.current_position
                    
                    # 应用PID控制
                    control_output = self._apply_pid_control(error, delta_time)
                    
                    # 发送控制命令
                    self._send_control_command(control_output, target_velocity)
                    
                    # 更新当前位置和速度
                    self.velocity = target_velocity
                    
                    # 检查运动是否完成
                    if elapsed_time >= self.motion_plan['total_time'] or abs(error) < 0.1:  # 位置误差小于阈值
                        self.current_position = self.target_position
                        self.velocity = 0.0
                        self.status = ActuatorStatus.IDLE
                        self.motion_plan = None
                        
                        # 通知运动完成
                        if self.motion_completed_callback:
                            try:
                                self.motion_completed_callback(self.actuator_id, self.current_position)
                            except Exception as e:
                                logger.error(f"运动完成回调出错 - ID: {self.actuator_id}, 错误: {str(e)}")
                                
                        logger.info(f"执行器运动完成 - ID: {self.actuator_id}, 位置: {self.current_position}")
                        self._notify_status_change()
                        
                    else:
                        # 估算当前位置
                        # 注意：实际应该使用传感器反馈的位置
                        self.current_position = target_position
                        
                        # 通知位置变更
                        if self.position_changed_callback:
                            try:
                                self.position_changed_callback(self.actuator_id, self.current_position)
                            except Exception as e:
                                logger.error(f"位置变更回调出错 - ID: {self.actuator_id}, 错误: {str(e)}")
                                
            except Exception as e:
                logger.error(f"控制循环出错 - ID: {self.actuator_id}, 错误: {str(e)}")
                self.status = ActuatorStatus.ERROR
                self.error_message = str(e)
                self._notify_status_change()
                
            # 控制频率
            time.sleep(0.01)  # 100Hz控制频率
            
    def _apply_pid_control(self, error: float, delta_time: float) -> float:
        """应用PID控制算法"""
        # 比例项
        p_term = self.pid_params['p'] * error
        
        # 积分项
        self.pid_integral += error * delta_time
        # 积分饱和防止
        integral_limit = self.config.get('pid_integral_limit', 10.0)
        self.pid_integral = max(-integral_limit, min(self.pid_integral, integral_limit))
        i_term = self.pid_params['i'] * self.pid_integral
        
        # 微分项
        derivative = (error - self.pid_last_error) / delta_time if delta_time > 0 else 0
        d_term = self.pid_params['d'] * derivative
        
        # 保存上次误差
        self.pid_last_error = error
        
        # 计算总输出
        output = p_term + i_term + d_term
        
        # 限制输出范围
        output_limit = self.config.get('output_limit', 100.0)
        output = max(-output_limit, min(output, output_limit))
        
        return output
        
    def _send_control_command(self, control_output: float, velocity: float):
        """发送控制命令到执行器"""
        if not self.device_interface:
            # 模拟控制
            logger.debug(f"模拟控制输出 - ID: {self.actuator_id}, 输出: {control_output}, 速度: {velocity}")
            return
            
        try:
            # 构建控制命令
            command = {
                'command': 'control',
                'actuator_id': self.actuator_id,
                'output': control_output,
                'velocity': velocity,
                'timestamp': time.time()
            }
            
            # 发送命令
            self.device_interface.send_data(command)
            
        except Exception as e:
            logger.error(f"发送控制命令失败 - ID: {self.actuator_id}, 错误: {str(e)}")
            
    def _send_stop_command(self):
        """发送停止命令到执行器"""
        if not self.device_interface:
            logger.debug(f"模拟停止命令 - ID: {self.actuator_id}")
            return
            
        try:
            # 构建停止命令
            command = {
                'command': 'stop',
                'actuator_id': self.actuator_id,
                'timestamp': time.time()
            }
            
            # 发送命令
            self.device_interface.send_data(command)
            
        except Exception as e:
            logger.error(f"发送停止命令失败 - ID: {self.actuator_id}, 错误: {str(e)}")
            
    def _read_sensors(self):
        """读取传感器数据"""
        # 这里应该从实际传感器读取数据
        # 简化实现，模拟传感器数据
        for sensor_id, sensor_config in self.sensors.items():
            # 模拟传感器数据
            # 实际应用中应该从设备接口或其他途径获取真实传感器数据
            self.sensor_readings[sensor_id] = {
                'value': np.random.normal(sensor_config.get('default_value', 0), sensor_config.get('noise', 0.1)),
                'timestamp': time.time()
            }
            
        # 通知传感器数据变更
        if self.sensor_data_callback:
            try:
                self.sensor_data_callback(self.actuator_id, self.sensor_readings.copy())
            except Exception as e:
                logger.error(f"传感器数据回调出错 - ID: {self.actuator_id}, 错误: {str(e)}")
                
    def _check_blocked(self) -> bool:
        """检查执行器是否被阻挡"""
        # 简化实现
        # 实际应用中应该根据传感器数据判断是否被阻挡
        if 'torque_sensor' in self.sensor_readings:
            torque_value = self.sensor_readings['torque_sensor']['value']
            torque_threshold = self.config.get('torque_threshold', 10.0)
            return abs(torque_value) > torque_threshold
        
        return False
        
    def _calibrate_sensors(self):
        """校准传感器"""
        # 简化实现
        logger.info(f"校准传感器 - ID: {self.actuator_id}, 传感器数量: {len(self.sensors)}")
        
    def _handle_device_data(self, device_id: str, data: Any):
        """处理从设备接收到的数据"""
        # 实际应用中应该根据设备返回的数据更新执行器状态
        logger.debug(f"接收到设备数据 - ID: {device_id}, 数据: {data}")
        
    def _notify_position_change(self):
        """通知位置变更"""
        if self.position_changed_callback:
            try:
                self.position_changed_callback(self.actuator_id, self.current_position)
            except Exception as e:
                logger.error(f"位置变更通知出错 - ID: {self.actuator_id}, 错误: {str(e)}")
                
    def _notify_status_change(self):
        """通知状态变更"""
        if self.status_changed_callback:
            try:
                self.status_changed_callback(self.actuator_id, self.get_status())
            except Exception as e:
                logger.error(f"状态变更通知出错 - ID: {self.actuator_id}, 错误: {str(e)}")

class StepperMotorController(ActuatorController):
    """步进电机控制器"""
    def __init__(self, actuator_id: str, config: Dict[str, Any]):
        # 默认步进电机配置
        default_config = {
            'steps_per_rev': 200,  # 每转步数
            'microstepping': 1,     # 微步数
            'max_velocity': 1000,   # 最大速度 (steps/s)
            'acceleration': 5000,   # 加速度 (steps/s²)
            'hold_torque': 0.5,     # 保持扭矩 (A)
            'run_current': 0.8      # 运行电流 (A)
        }
        
        # 合并配置
        merged_config = {**default_config, **config}
        
        super().__init__(actuator_id, ActuatorType.STEPPER, merged_config)
        
    def _initialize_actuator(self):
        """初始化步进电机"""
        logger.info(f"初始化步进电机 - ID: {self.actuator_id}")
        
        # 设置电流参数
        if self.device_interface:
            try:
                current_command = {
                    'command': 'set_current',
                    'hold_current': self.config['hold_torque'],
                    'run_current': self.config['run_current']
                }
                self.device_interface.send_data(current_command)
            except Exception as e:
                logger.warning(f"设置电流参数失败 - ID: {self.actuator_id}, 错误: {str(e)}")
                
        # 计算每毫米步数（如果是线性执行器）
        if 'mm_per_rev' in self.config:
            self.steps_per_mm = (self.config['steps_per_rev'] * self.config['microstepping']) / self.config['mm_per_rev']
        else:
            self.steps_per_mm = 1.0

class ServoMotorController(ActuatorController):
    """伺服电机控制器"""
    def __init__(self, actuator_id: str, config: Dict[str, Any]):
        # 默认伺服电机配置
        default_config = {
            'min_pulse_width': 500,    # 最小脉冲宽度 (us)
            'max_pulse_width': 2500,   # 最大脉冲宽度 (us)
            'min_angle': 0,            # 最小角度 (deg)
            'max_angle': 180,          # 最大角度 (deg)
            'neutral_pulse': 1500,     # 中立点脉冲宽度 (us)
            'response_time': 0.1       # 响应时间 (s)
        }
        
        # 合并配置并设置位置限制
        merged_config = {
            **default_config,
            **config,
            'min_position': default_config['min_angle'],
            'max_position': default_config['max_angle']
        }
        
        super().__init__(actuator_id, ActuatorType.SERVO, merged_config)
        
    def _initialize_actuator(self):
        """初始化伺服电机"""
        logger.info(f"初始化伺服电机 - ID: {self.actuator_id}")
        
        # 设置初始位置到中立点
        if self.device_interface:
            try:
                neutral_command = {
                    'command': 'set_neutral',
                    'pulse_width': self.config['neutral_pulse']
                }
                self.device_interface.send_data(neutral_command)
            except Exception as e:
                logger.warning(f"设置中立点失败 - ID: {self.actuator_id}, 错误: {str(e)}")
                
    def _send_control_command(self, control_output: float, velocity: float):
        """发送控制命令到伺服电机"""
        # 将控制输出转换为脉冲宽度
        angle = self.current_position  # 伺服电机位置以角度表示
        
        # 计算脉冲宽度
        pulse_width = self.config['min_pulse_width'] + \
                     (self.config['max_pulse_width'] - self.config['min_pulse_width']) * \
                     (angle - self.config['min_angle']) / \
                     (self.config['max_angle'] - self.config['min_angle'])
        
        pulse_width = max(self.config['min_pulse_width'], min(pulse_width, self.config['max_pulse_width']))
        
        if not self.device_interface:
            # 模拟控制
            logger.debug(f"模拟伺服控制 - ID: {self.actuator_id}, 角度: {angle}°, 脉冲宽度: {pulse_width}us")
            return
            
        try:
            # 构建控制命令
            command = {
                'command': 'set_position',
                'actuator_id': self.actuator_id,
                'pulse_width': pulse_width,
                'angle': angle,
                'timestamp': time.time()
            }
            
            # 发送命令
            self.device_interface.send_data(command)
            
        except Exception as e:
            logger.error(f"发送伺服控制命令失败 - ID: {self.actuator_id}, 错误: {str(e)}")

class ActuatorControllerManager:
    """\执行器控制器管理器，管理所有执行器控制器"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ActuatorControllerManager, cls).__new__(cls)
            cls._instance.controllers = {}
            cls._instance.lock = threading.Lock()
        return cls._instance
        
    def create_actuator_controller(self, actuator_id: str, actuator_type: Union[str, ActuatorType], config: Dict[str, Any]) -> ActuatorController:
        """
        创建执行器控制器
        
        参数:
            actuator_id: 执行器ID
            actuator_type: 执行器类型
            config: 配置参数
        
        返回:
            执行器控制器实例
        """
        with self.lock:
            if actuator_id in self.controllers:
                logger.warning(f"执行器ID已存在，将返回已有控制器 - ID: {actuator_id}")
                return self.controllers[actuator_id]
                
            # 转换类型为枚举
            if isinstance(actuator_type, str):
                try:
                    type_enum = ActuatorType(actuator_type.lower())
                except ValueError:
                    raise ActuatorControlException(f"不支持的执行器类型: {actuator_type}")
            else:
                type_enum = actuator_type
                
            # 根据类型创建控制器
            if type_enum == ActuatorType.STEPPER:
                controller = StepperMotorController(actuator_id, config)
            elif type_enum == ActuatorType.SERVO:
                controller = ServoMotorController(actuator_id, config)
            else:
                # 对于其他类型，使用基类（实际应用中应该创建对应的子类）
                controller = ActuatorController(actuator_id, type_enum, config)
                
            # 保存控制器
            self.controllers[actuator_id] = controller
            
            logger.info(f"执行器控制器创建成功 - ID: {actuator_id}, 类型: {type_enum.value}")
            return controller
            
    def get_actuator_controller(self, actuator_id: str) -> Optional[ActuatorController]:
        """获取执行器控制器"""
        with self.lock:
            return self.controllers.get(actuator_id)
            
    def initialize_actuator(self, actuator_id: str) -> bool:
        """初始化执行器"""
        controller = self.get_actuator_controller(actuator_id)
        if controller:
            return controller.initialize()
        logger.error(f"执行器控制器不存在 - ID: {actuator_id}")
        return False
        
    def shutdown_actuator(self, actuator_id: str) -> bool:
        """关闭执行器"""
        controller = self.get_actuator_controller(actuator_id)
        if controller:
            return controller.shutdown()
        logger.error(f"执行器控制器不存在 - ID: {actuator_id}")
        return False
        
    def move_actuator(self, actuator_id: str, position: float, velocity: Optional[float] = None) -> bool:
        """移动执行器到指定位置"""
        controller = self.get_actuator_controller(actuator_id)
        if controller:
            return controller.move_to(position, velocity)
        logger.error(f"执行器控制器不存在 - ID: {actuator_id}")
        return False
        
    def move_actuator_relative(self, actuator_id: str, distance: float, velocity: Optional[float] = None) -> bool:
        """相对移动执行器"""
        controller = self.get_actuator_controller(actuator_id)
        if controller:
            return controller.move_relative(distance, velocity)
        logger.error(f"执行器控制器不存在 - ID: {actuator_id}")
        return False
        
    def stop_actuator(self, actuator_id: str) -> bool:
        """停止执行器"""
        controller = self.get_actuator_controller(actuator_id)
        if controller:
            return controller.stop()
        logger.error(f"执行器控制器不存在 - ID: {actuator_id}")
        return False
        
    def get_actuator_status(self, actuator_id: str) -> Optional[Dict[str, Any]]:
        """获取执行器状态"""
        controller = self.get_actuator_controller(actuator_id)
        if controller:
            return controller.get_status()
        logger.error(f"执行器控制器不存在 - ID: {actuator_id}")
        return None
        
    def get_all_actuators_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有执行器状态"""
        statuses = {}
        with self.lock:
            for actuator_id, controller in self.controllers.items():
                statuses[actuator_id] = controller.get_status()
        return statuses
        
    def read_sensors(self, actuator_id: str) -> Optional[Dict[str, Any]]:
        """读取执行器的传感器数据"""
        controller = self.get_actuator_controller(actuator_id)
        if controller:
            return controller.read_sensors()
        logger.error(f"执行器控制器不存在 - ID: {actuator_id}")
        return None
        
    def remove_actuator_controller(self, actuator_id: str) -> bool:
        """移除执行器控制器"""
        with self.lock:
            if actuator_id in self.controllers:
                # 关闭执行器
                self.controllers[actuator_id].shutdown()
                # 移除控制器
                del self.controllers[actuator_id]
                logger.info(f"执行器控制器已移除 - ID: {actuator_id}")
                return True
            logger.error(f"执行器控制器不存在 - ID: {actuator_id}")
            return False
            
# 创建全局实例
actuator_controller_manager = ActuatorControllerManager()