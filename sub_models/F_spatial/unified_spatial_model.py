# -*- coding: utf-8 -*-
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0

"""
统一空间定位模型 | Unified Spatial Positioning Model
整合标准模式和增强模式功能
"""

import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass

@dataclass
class Position:
    """位置数据结构"""
    x: float
    y: float
    z: float = 0.0
    timestamp: float = None
    accuracy: float = 1.0

@dataclass
class SpatialMap:
    """空间地图数据结构"""
    id: str
    positions: List[Position]
    metadata: Dict[str, Any]
    created_at: float

class UnifiedSpatialModel:
    """
    统一空间定位模型
    支持位置计算、路径规划、空间映射等功能
    """
    
    def __init__(self, mode: str = "standard", config: Optional[Dict] = None):
        self.mode = mode
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 空间数据
        self.current_position = Position(0, 0, 0)
        self.position_history = []
        self.spatial_maps = {}
        self.waypoints = []
        
        # 配置参数
        self.accuracy_threshold = self.config.get("accuracy_threshold", 0.5)
        self.max_history_size = self.config.get("max_history_size", 1000)
        
    def update_position(self, x: float, y: float, z: float = 0.0, 
                       accuracy: float = 1.0) -> Position:
        """更新当前位置"""
        import time
        
        new_position = Position(x, y, z, time.time(), accuracy)
        self.current_position = new_position
        
        # 记录历史
        self.position_history.append(new_position)
        if len(self.position_history) > self.max_history_size:
            self.position_history.pop(0)
        
        # 增强模式下的额外处理
        if self.mode == "enhanced":
            new_position = self._enhanced_position_processing(new_position)
        
        return new_position
    
    def calculate_distance(self, pos1: Position, pos2: Position) -> float:
        """计算两点间距离"""
        return math.sqrt(
            (pos2.x - pos1.x)**2 + 
            (pos2.y - pos1.y)**2 + 
            (pos2.z - pos1.z)**2
        )
    
    def calculate_bearing(self, pos1: Position, pos2: Position) -> float:
        """计算方位角（度）"""
        angle = math.atan2(pos2.y - pos1.y, pos2.x - pos1.x)
        return math.degrees(angle)
    
    def create_waypoint(self, x: float, y: float, z: float = 0.0, 
                       label: str = "") -> Dict[str, Any]:
        """创建路径点"""
        waypoint = {
            "position": Position(x, y, z),
            "label": label,
            "created_at": time.time(),
            "visited": False
        }
        self.waypoints.append(waypoint)
        return waypoint
    
    def plan_path(self, start: Position, end: Position, 
                  obstacles: List[Position] = None) -> List[Position]:
        """路径规划"""
        if obstacles is None:
            obstacles = []
        
        if self.mode == "enhanced" and obstacles:
            # 增强模式：使用A*算法
            return self._astar_pathfinding(start, end, obstacles)
        else:
            # 标准模式：直线
            return [start, end]
    
    def create_spatial_map(self, map_id: str, positions: List[Position], 
                          metadata: Dict[str, Any] = None) -> SpatialMap:
        """创建空间地图"""
        spatial_map = SpatialMap(
            id=map_id,
            positions=positions,
            metadata=metadata or {},
            created_at=time.time()
        )
        self.spatial_maps[map_id] = spatial_map
        return spatial_map
    
    def get_position_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取位置历史"""
        return [
            {
                "x": pos.x,
                "y": pos.y,
                "z": pos.z,
                "timestamp": pos.timestamp,
                "accuracy": pos.accuracy
            }
            for pos in self.position_history[-limit:]
        ]
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "current_position": {
                "x": self.current_position.x,
                "y": self.current_position.y,
                "z": self.current_position.z,
                "accuracy": self.current_position.accuracy
            },
            "history_count": len(self.position_history),
            "waypoints_count": len(self.waypoints),
            "maps_count": len(self.spatial_maps),
            "mode": self.mode
        }
    
    def _enhanced_position_processing(self, position: Position) -> Position:
        """增强模式下的位置处理"""
        # 平滑处理
        if len(self.position_history) >= 3:
            recent_positions = self.position_history[-3:]
            avg_x = np.mean([p.x for p in recent_positions])
            avg_y = np.mean([p.y for p in recent_positions])
            avg_z = np.mean([p.z for p in recent_positions])
            
            # 应用平滑
            position.x = position.x * 0.7 + avg_x * 0.3
            position.y = position.y * 0.7 + avg_y * 0.3
            position.z = position.z * 0.7 + avg_z * 0.3
        
        return position
    
    def _astar_pathfinding(self, start: Position, end: Position, 
                        obstacles: List[Position]) -> List[Position]:
        """A*路径规划算法"""
        # 简化的A*实现
        path = [start]
        
        # 检查是否有障碍物
        obstacle_positions = [(obs.x, obs.y) for obs in obstacles]
        
        # 生成路径点
        steps = max(10, int(self.calculate_distance(start, end) / 0.5))
        
        for i in range(1, steps + 1):
            t = i / steps
            x = start.x + (end.x - start.x) * t
            y = start.y + (end.y - start.y) * t
            z = start.z + (end.z - start.z) * t
            
            # 检查障碍物
            if (x, y) not in obstacle_positions:
                path.append(Position(x, y, z))
        
        return path
    
    def calculate_area(self, positions: List[Position]) -> float:
        """计算多边形面积"""
        if len(positions) < 3:
            return 0.0
        
        # 使用鞋带公式
        n = len(positions)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += positions[i].x * positions[j].y
            area -= positions[j].x * positions[i].y
        
        return abs(area) / 2.0