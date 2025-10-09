# -*- coding: utf-8 -*-
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0

"""
Unified Motion Control Model
Integrates standard and enhanced mode functionality
"""

import threading
import time
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json

@dataclass
class MotionCommand:
    """Motion command data structure"""
    action: str
    parameters: Dict[str, Any]
    speed: float = 1.0
    duration: float = None
    priority: int = 1

@dataclass
class MotionState:
    """Motion state data structure"""
    position: Dict[str, float]
    velocity: Dict[str, float]
    acceleration: Dict[str, float]
    timestamp: float
    is_moving: bool

class UnifiedMotionModel:
    """
    Unified Motion Control Model
    Supports multi-axis motion control, trajectory planning, real-time feedback and other functions
    """
    
    def __init__(self, mode: str = "standard", config: Optional[Dict] = None):
        self.mode = mode
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Motion parameters
        self.axes = self.config.get("axes", ["x", "y", "z"])
        self.max_speed = self.config.get("max_speed", 100.0)
        self.max_acceleration = self.config.get("max_acceleration", 50.0)
        
        # Current state
        self.current_position = {axis: 0.0 for axis in self.axes}
        self.current_velocity = {axis: 0.0 for axis in self.axes}
        self.current_acceleration = {axis: 0.0 for axis in self.axes}
        
        # Motion control
        self.is_moving = False
        self.motion_queue = []
        self.motion_thread = None
        self.emergency_stop = False
        
        # History records
        self.motion_history = []
        self.max_history = 1000
        
        # Thread safety
        self._lock = threading.Lock()
        
    def move_to(self, target_position: Dict[str, float], 
                speed: float = None, interpolation: str = "linear") -> bool:
        """Move to specified position"""
        if self.emergency_stop:
            return False
        
        with self._lock:
            # Validate target position
            for axis, value in target_position.items():
                if axis not in self.axes:
                    return False
            
            # Create motion command
            command = MotionCommand(
                action="move_to",
                parameters={"target": target_position, "interpolation": interpolation},
                speed=speed or self.max_speed
            )
            
            self.motion_queue.append(command)
            
            if not self.is_moving:
                self._start_motion_control()
            
            return True
    
    def move_relative(self, delta_position: Dict[str, float], 
                     speed: float = None) -> bool:
        """Relative movement"""
        target_position = {}
        for axis, delta in delta_position.items():
            if axis in self.current_position:
                target_position[axis] = self.current_position[axis] + delta
        
        return self.move_to(target_position, speed)
    
    def home_all_axes(self) -> bool:
        """Home all axes"""
        home_position = {axis: 0.0 for axis in self.axes}
        return self.move_to(home_position)
    
    def emergency_stop_all(self):
        """Emergency stop"""
        self.emergency_stop = True
        self.is_moving = False
        self.motion_queue.clear()
        
        # Stop all axes
        for axis in self.axes:
            self.current_velocity[axis] = 0.0
            self.current_acceleration[axis] = 0.0
        
        self.logger.warning("Emergency stop triggered")
    
    def reset_emergency_stop(self):
        """Reset emergency stop"""
        self.emergency_stop = False
        self.logger.info("Emergency stop reset")
    
    def get_current_position(self) -> Dict[str, float]:
        """Get current position"""
        with self._lock:
            return self.current_position.copy()
    
    def get_motion_state(self) -> MotionState:
        """Get motion state"""
        with self._lock:
            return MotionState(
                position=self.current_position.copy(),
                velocity=self.current_velocity.copy(),
                acceleration=self.current_acceleration.copy(),
                timestamp=time.time(),
                is_moving=self.is_moving
            )
    
    def set_axis_limits(self, axis: str, min_limit: float, max_limit: float) -> bool:
        """Set axis limits"""
        if axis not in self.axes:
            return False
        
        self.config.setdefault("axis_limits", {})[axis] = {
            "min": min_limit,
            "max": max_limit
        }
        return True
    
    def execute_trajectory(self, trajectory: List[Dict[str, float]], 
                          speed: float = None) -> bool:
        """Execute trajectory"""
        if self.emergency_stop:
            return False
        
        for point in trajectory:
            if not self.move_to(point, speed):
                return False
        
        return True
    
    def get_motion_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get motion history"""
        with self._lock:
            return self.motion_history[-limit:]
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current status"""
        state = self.get_motion_state()
        return {
            "position": state.position,
            "velocity": state.velocity,
            "acceleration": state.acceleration,
            "is_moving": state.is_moving,
            "queue_length": len(self.motion_queue),
            "history_length": len(self.motion_history),
            "emergency_stop": self.emergency_stop,
            "mode": self.mode,
            "axes": self.axes
        }
    
    def _start_motion_control(self):
        """Start motion control thread"""
        if self.motion_thread is None or not self.motion_thread.is_alive():
            self.motion_thread = threading.Thread(target=self._motion_control_loop)
            self.motion_thread.daemon = True
            self.motion_thread.start()
    
    def _motion_control_loop(self):
        """Motion control loop"""
        while True:
            with self._lock:
                if not self.motion_queue or self.emergency_stop:
                    self.is_moving = False
                    break
                
                command = self.motion_queue.pop(0)
                self.is_moving = True
            
            try:
                self._execute_motion_command(command)
            except Exception as e:
                self.logger.error(f"Motion command execution failed: {e}")
                break
    
    def _execute_motion_command(self, command: MotionCommand):
        """Execute motion command"""
        if command.action == "move_to":
            target = command.parameters["target"]
            interpolation = command.parameters.get("interpolation", "linear")
            
            # Calculate move time
            max_distance = 0
            for axis, target_pos in target.items():
                if axis in self.current_position:
                    distance = abs(target_pos - self.current_position[axis])
                    max_distance = max(max_distance, distance)
            
            move_time = max_distance / command.speed
            
            # Execute move
            steps = max(10, int(move_time * 10))
            for i in range(steps + 1):
                if self.emergency_stop:
                    break
                
                t = i / steps
                
                with self._lock:
                    # Interpolation calculation
                    for axis, target_pos in target.items():
                        if axis in self.current_position:
                            start_pos = self.current_position[axis]
                            
                            if interpolation == "linear":
                                new_pos = start_pos + (target_pos - start_pos) * t
                            else:
                                # Smooth interpolation
                                smooth_t = 0.5 - 0.5 * np.cos(np.pi * t)
                                new_pos = start_pos + (target_pos - start_pos) * smooth_t
                            
                            self.current_position[axis] = new_pos
                            self.current_velocity[axis] = (target_pos - start_pos) / move_time
                    
                    # Record history
                    self.motion_history.append({
                        "timestamp": time.time(),
                        "position": self.current_position.copy(),
                        "velocity": self.current_velocity.copy(),
                        "command": command.__dict__
                    })
                    
                    if len(self.motion_history) > self.max_history:
                        self.motion_history.pop(0)
                
                time.sleep(0.1)
            
            # Stop motion
            with self._lock:
                for axis in self.current_velocity:
                    self.current_velocity[axis] = 0.0
