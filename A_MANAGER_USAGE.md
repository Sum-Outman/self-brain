# A Management Model API 使用指南

## 🚀 快速启动

### 方法1：直接启动（推荐）
```bash
python a_manager_standalone.py
```

### 方法2：使用启动脚本（Windows）
双击运行： `start_a_manager.bat`

## 📍 服务地址
- **主地址**: http://localhost:5014
- **本地地址**: http://127.0.0.1:5014

## 🔧 可用API端点

### 1. 健康检查
```bash
curl http://localhost:5014/api/health
```

### 2. 获取模型列表
```bash
curl http://localhost:5014/api/models
```

### 3. 处理消息（核心功能）
```bash
curl -X POST http://localhost:5014/process_message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you?",
    "task_type": "general"
  }'
```

**支持的任务类型**:
- `general` - 通用对话
- `programming` - 编程问题
- `knowledge` - 知识查询
- `creative` - 创意内容
- `analysis` - 分析任务

### 4. 情感分析
```bash
curl -X POST http://localhost:5014/api/emotion/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I love this beautiful day!"
  }'
```

### 5. 系统统计
```bash
curl http://localhost:5014/api/system/stats
```

## 🎯 Python使用示例

```python
import requests

# 基础使用
url = "http://localhost:5003/process_message"
payload = {
    "message": "How do I create a Python class?",
    "task_type": "programming"
}

response = requests.post(url, json=payload)
result = response.json()
print("回复:", result["response"])
print("任务ID:", result["task_id"])
```

## 📋 模型列表
当前系统包含11个AI模型：
1. A_management - 主管理模型
2. B_language - 语言处理
3. C_vision - 视觉处理
4. D_audio - 音频处理
5. E_reasoning - 推理模型
6. F_emotion - 情感分析
7. G_sensor - 传感器
8. H_computer_control - 计算机控制
9. I_knowledge - 知识库
10. J_motion - 运动控制
11. K_programming - 编程助手

## ✅ 测试验证
所有端点都已通过测试验证，可正常使用。

## 🔄 重启服务
如果需要重启服务，请：
1. 按 `Ctrl+C` 停止当前服务
2. 重新运行启动命令