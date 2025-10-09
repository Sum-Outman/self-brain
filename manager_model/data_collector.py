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

import asyncio
import psutil
from sub_models import (
    B_language, C_audio, D_image, E_video, 
    F_spatial, G_sensor, H_computer_control,
    I_knowledge, J_motion, K_programming
)

async def collect_real_time_data():
    """收集系统实时数据"""
    data = {
        "models": {},
        "system": {},
        "sensors": {},
        "knowledge": {},
        "inputs": {}
    }
    
    # 收集模型状态
    models = {
        "A": {"name": "管理模型", "module": None},
        "B": {"name": "大语言模型", "module": B_language},
        "C": {"name": "音频处理模型", "module": C_audio},
        "D": {"name": "图片视觉处理模型", "module": D_image},
        "E": {"name": "视频流视觉处理模型", "module": E_video},
        "F": {"name": "双目空间定位感知模型", "module": F_spatial},
        "G": {"name": "传感器感知模型", "module": G_sensor},
        "H": {"name": "计算机控制模型", "module": H_computer_control},
        "I": {"name": "知识库专家模型", "module": I_knowledge},
        "J": {"name": "运动和执行器控制模型", "module": J_motion},
        "K": {"name": "编程模型", "module": K_programming}
    }
    
    for model_id, info in models.items():
        status = "idle"
        cpu = 0
        memory = 0
        
        if info["module"]:
            try:
                status = await info["module"].get_status()
                cpu = await info["module"].get_cpu_usage()
                memory = await info["module"].get_memory_usage()
            except Exception as e:
                print(f"获取模型 {model_id} 状态失败: {str(e)}")
        
        data["models"][model_id] = {
            "status": status,
            "cpu": cpu,
            "memory": memory
        }
    
    # 收集系统指标
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    net_io = psutil.net_io_counters()
    
    data["system"] = {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "network_in": net_io.bytes_recv,
        "network_out": net_io.bytes_sent,
        "active_tasks": len([m for m in data["models"].values() if m["status"] == "active"])
    }
    
    # 收集传感器数据
    try:
        sensor_data = await G_sensor.get_sensor_data()
        data["sensors"] = {
            "temperature": sensor_data.get("temperature", 0),
            "humidity": sensor_data.get("humidity", 0),
            "pressure": sensor_data.get("pressure", 0),
            "distance": sensor_data.get("distance", 0)
        }
    except Exception as e:
        print(f"获取传感器数据失败: {str(e)}")
    
    # 收集知识库指标
    try:
        knowledge_stats = await I_knowledge.get_knowledge_stats()
        data["knowledge"] = knowledge_stats
    except Exception as e:
        print(f"获取知识库指标失败: {str(e)}")
    
    # 收集输入数据统计
    try:
        input_stats = await B_language.get_input_stats()
        data["inputs"] = {
            "audio": input_stats.get("audio", 0),
            "video": input_stats.get("video", 0),
            "text": input_stats.get("text", 0),
            "sensor": input_stats.get("sensor", 0)
        }
    except Exception as e:
        print(f"获取输入统计失败: {str(e)}")
    
    return data
