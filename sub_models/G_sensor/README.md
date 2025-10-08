# 传感器集成与外接设备通讯模块

## Sensor Integration and External Device Communication Module

该模块负责将各种传感器数据集成到主系统，并提供与外接设备的通讯接口，实现实时数据采集、处理、异常检测和多传感器数据融合等功能。

## 功能概述

- **多类型传感器支持**：温度、湿度、加速度、陀螺仪、压力、光线等多种传感器
- **实时数据处理**：支持摄像头、麦克风、网络流和串口的实时数据采集和处理
- **异常检测**：基于阈值和历史数据分析的异常检测功能
- **数据融合**：多传感器数据加权融合算法
- **外部API集成**：支持OpenAI、Azure、Google Cloud和AWS等外部API集成
- **RESTful API**：提供标准化的HTTP接口供其他模块调用
- **数据总线连接**：通过Redis与主模型进行数据交换
- **多语言支持**：支持中英文切换

## 安装与配置

### 环境要求

- Python 3.8+ 
- 依赖包：
  - torch (PyTorch)
  - opencv-python (OpenCV)
  - redis
  - pyaudio
  - pyserial
  - flask
  - flask-cors
  - requests
  - numpy

### 快速启动

使用提供的启动脚本快速启动传感器服务：

```bash
# Windows
cd D:\shiyan\sub_models\G_sensor
start_sensor_service.bat

# Linux/Mac
cd /path/to/shiyan/sub_models/G_sensor
python sensor_integration.py
```

### 配置文件

主要配置文件位于 `config/sensor_config.json`，包含以下关键配置项：

- **基本配置**：模型类型、默认语言、数据总线连接信息
- **实时接口配置**：摄像头、麦克风、网络流和串口的配置
- **外部API配置**：各外部API的启用状态、API密钥和端点
- **传感器配置**：各类型传感器的单位、范围、精度、采样率和异常阈值
- **通信协议配置**：支持的通信协议参数

示例配置片段：

```json
{
  "model_type": "local",
  "default_language": "zh",
  "data_bus_host": "localhost",
  "data_bus_port": 6379,
  "realtime_interfaces": {
    "camera": {"enabled": true, "device_index": 0},
    "microphone": {"enabled": true, "device_index": 0},
    "network_streams": {"enabled": true},
    "serial_ports": {"enabled": true}
  }
}
```

## API接口文档

传感器模块提供以下RESTful API接口：

### 1. 处理传感器数据

```
POST /api/sensor/process
```

**请求体**：传感器数据JSON对象，例如：
```json
{
  "temperature": 25.5,
  "humidity": 60.0,
  "acceleration": 0.5
}
```

**响应**：处理结果，包含各传感器的状态、置信度等信息

### 2. 检测传感器异常

```
POST /api/sensor/detect_anomalies
```

**请求体**：传感器数据JSON对象，可选`use_history`参数
```json
{
  "temperature": 150.0,
  "humidity": 60.0,
  "use_history": true
}
```

**响应**：异常检测结果，包含异常列表、异常数量和置信度

### 3. 融合多传感器数据

```
POST /api/sensor/fuse
```

**请求体**：传感器数据列表，可选`confidence`权重参数
```json
[
  {"temperature": 25.0, "humidity": 59.0, "confidence": 0.9},
  {"temperature": 26.0, "humidity": 61.0, "confidence": 0.8}
]
```

**响应**：融合结果，包含加权平均后的数据和使用的权重

### 4. 获取传感器状态

```
GET /api/sensor/status
```

**响应**：传感器系统状态，包含接口状态和配置概览

### 5. 使用外部API处理数据

```
POST /api/sensor/external_api/process
```

**请求体**：包含API名称和要处理的数据
```json
{
  "api_name": "openai",
  "data": "分析这段传感器数据：温度25.5°C，湿度60%"
}
```

**响应**：外部API处理结果

## 客户端使用示例

使用提供的`sensor_api_client.py`示例程序可以快速测试和使用传感器API：

```bash
python sensor_api_client.py
```

该示例程序演示了如何：
- 获取传感器状态
- 处理单组传感器数据
- 检测异常数据
- 融合多传感器数据
- 连续采集和监控传感器数据

## 实时数据处理

传感器模块支持以下实时数据源的采集和处理：

1. **摄像头**：通过OpenCV采集视频流，进行亮度、对比度和边缘密度分析
2. **麦克风**：通过PyAudio采集音频流，进行音量和振幅分析
3. **网络流**：支持RTSP、RTMP、HTTP等协议的网络视频流
4. **串口设备**：支持RS232/RS485等串口设备的数据读写

实时数据处理默认在服务启动时自动启动，可以通过配置文件控制各接口的启用状态。

## 外部API集成

传感器模块支持集成以下外部API服务：

1. **OpenAI**：用于文本分析和处理
2. **Azure Computer Vision**：用于图像分析
3. **Google Cloud Vision**：用于图像标签识别
4. **AWS Rekognition**：用于图像内容分析

外部API需要在配置文件中启用并提供相应的认证信息。

## 数据总线集成

传感器模块通过Redis与主模型进行数据交换：

- 发布传感器数据到`sensor_data`频道
- 支持订阅其他模块发布的数据

确保Redis服务正在运行，并且配置文件中的连接信息正确。

## 开发与扩展

### 添加新传感器类型

1. 在`config/sensor_config.json`中的`sensor_configuration`部分添加新传感器的配置
2. 确保配置包含单位、范围、精度、采样率和异常阈值等信息

### 添加新的通信协议

1. 在`config/sensor_config.json`中的`communication_protocols`部分添加新协议配置
2. 在`SensorModel`类中实现相应的连接、读写方法

### 集成新的外部API

1. 在`config/sensor_config.json`中的`external_api.apis`部分添加新API配置
2. 在`connect_external_api`和`process_with_external_api`方法中添加相应的处理逻辑

## 故障排除

### 常见问题

1. **服务无法启动**
   - 检查Python和依赖包是否正确安装
   - 检查端口是否被占用（默认端口5006）
   - 查看日志文件了解详细错误信息

2. **无法连接数据总线**
   - 确认Redis服务是否正在运行
   - 检查配置文件中的连接信息是否正确

3. **实时接口无法工作**
   - 确认设备是否正确连接
   - 检查设备驱动是否安装
   - 检查配置文件中的设备索引是否正确

4. **外部API调用失败**
   - 确认API密钥和端点配置是否正确
   - 检查网络连接是否正常
   - 查看日志了解详细错误信息

## 日志

传感器模块的日志默认保存在`logs/sensor_integration.log`文件中，可以通过修改日志配置调整日志级别和输出位置。

## 注意事项

- 确保在生产环境中修改默认的安全配置，特别是API密钥和认证信息
- 对于实时数据处理，建议使用足够性能的硬件以确保流畅运行
- 定期清理历史数据以避免内存占用过高
- 在连接外部设备时，确保遵循相应的安全规范和操作规程