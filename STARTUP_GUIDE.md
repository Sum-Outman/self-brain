# 🚀 Self Brain AGI System - Updated Startup Guide

## 🎯 Quick Start (Updated Ports - No Conflicts)

### 1. 启动独立A Manager (推荐)
```bash
# 启动独立的A Management Model
python a_manager_standalone.py

# 访问地址: http://localhost:5014
# 健康检查: http://localhost:5014/api/health
```

### 2. 启动Manager Model API
```bash
# 进入manager_model目录
cd manager_model
python app.py

# 访问地址: http://localhost:5015
# 健康检查: http://localhost:5015/api/health
```

### 3. 一键启动所有服务
```bash
# 使用新的启动脚本
start_system_updated.bat
```

## 📍 服务地址映射 (无冲突)

| Service | Port | URL | Status |
|---------|------|-----|--------|
| **A Management Model** | 5001 | http://localhost:5001 | ✅ 主系统 |
| **B Language Model** | 5002 | http://localhost:5002 | ✅ 语言处理 |
| **C Audio Model** | 5003 | http://localhost:5003 | ✅ 音频处理 |
| **D Image Model** | 5004 | http://localhost:5004 | ✅ 图像处理 |
| **E Video Model** | 5005 | http://localhost:5005 | ✅ 视频处理 |
| **F Spatial Model** | 5006 | http://localhost:5006 | ✅ 空间处理 |
| **G Sensor Model** | 5007 | http://localhost:5007 | ✅ 传感器 |
| **H Computer Control** | 5008 | http://localhost:5008 | ✅ 系统控制 |
| **I Knowledge Model** | 5009 | http://localhost:5009 | ✅ 知识库 |
| **J Motion Model** | 5010 | http://localhost:5010 | ✅ 运动控制 |
| **K Programming Model** | 5011 | http://localhost:5011 | ✅ 编程模型 |
| **Training Manager** | 5012 | http://localhost:5012 | ✅ 训练管理 |
| **Quantum Integration** | 5013 | http://localhost:5013 | ✅ 量子接口 |
| **Standalone A Manager** | 5014 | http://localhost:5014 | ✅ 独立版本 |
| **Manager Model API** | 5015 | http://localhost:5015 | ✅ 管理API |

## 🔧 测试命令

### 健康检查
```bash
# 检查所有服务
python check_ports.py

# 检查特定服务
curl http://localhost:5014/api/health
curl http://localhost:5015/api/health
```

### 功能测试
```bash
# 测试消息处理
curl -X POST http://localhost:5014/process_message \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello, test message"}'

# 测试模型列表
curl http://localhost:5014/api/models

# 测试情感分析
curl -X POST http://localhost:5014/api/emotion/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"I am very happy today!"}'
```

## 🛠️ 环境变量配置

### Windows (PowerShell)
```powershell
$env:PORT_A_MANAGER=5014
python a_manager_standalone.py
```

### Windows (CMD)
```cmd
set PORT_A_MANAGER=5014
python a_manager_standalone.py
```

### Linux/Mac
```bash
export PORT_A_MANAGER=5014
python a_manager_standalone.py
```

## 🎯 启动推荐

1. **开发测试**: 使用独立版本 `a_manager_standalone.py` (端口5014)
2. **完整系统**: 使用 `start_system_updated.bat` 启动所有服务
3. **API开发**: 使用 `manager_model/app.py` (端口5015)

## ✅ 验证步骤

1. 运行 `python check_ports.py` 确认端口可用
2. 启动对应服务
3. 访问健康检查端点验证服务状态
4. 使用提供的测试命令验证功能

## 📋 故障排除

### 端口被占用
```bash
# 检查端口占用
netstat -ano | findstr :5014

# 使用备用端口
set PORT_A_MANAGER=5016
python a_manager_standalone.py
```

### 服务无法启动
```bash
# 检查依赖
pip install -r requirements.txt

# 查看日志
python a_manager_standalone.py > debug.log 2>&1
```