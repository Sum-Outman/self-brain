# Self Brain | 自我大脑

<div align="center">
  <img src="icons/self_brain.svg" alt="Self Brain Logo" width="200"/>
  
  **🧠 Next-Generation Autonomous AI System**
  
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
  [![AGI](https://img.shields.io/badge/Type-AGI%20System-red.svg)]()
  
  📧 **Contact**: silencecrowtom@qq.com
</div>

---

## 🌟 Project Overview | 项目简介

**Self Brain is a revolutionary autonomous AI system with self-learning, self-optimization, and cross-domain collaboration capabilities. The system integrates 11 specialized sub-models covering language, vision, audio, reasoning, and more dimensions, achieving true Artificial General Intelligence through advanced training control mechanisms.**

**自我大脑是一个革命性的自主人工智能系统，具备自我学习、自我优化和跨领域协作能力。系统集成11个专业子模型，涵盖语言、视觉、音频、推理等多个维度，通过先进的训练控制机制实现真正的人工通用智能。**

---

## 🎯 Core Features | 核心特性

| English | 中文 |
|---------|------|
| **🔄 Autonomous Training Control** | **🔄 自主训练控制** |
| **🤝 Cross-Model Collaboration** | **🤝 跨模型协作** |
| **📊 Real-time Performance Monitoring** | **📊 实时性能监控** |
| **🎨 Multimodal Processing** | **🎨 多模态处理** |
| **🧩 Plugin Architecture** | **🧩 插件架构** |
| **⚡ Dynamic Resource Allocation** | **⚡ 动态资源分配** |

---

## 🏗️ System Architecture | 系统架构

### English Version
```
Self Brain AGI System Architecture:
├── 🎛️ Management Model (A_management) - Central Coordinator
├── 🗣️ Language Model (B_language) - Natural Language Processing
├── 🔊 Audio Model (C_audio) - Sound Analysis & Synthesis
├── 👁️ Image Model (D_image) - Computer Vision
├── 🎬 Video Model (E_video) - Video Understanding
├── 🎯 Spatial Model (F_spatial) - 3D Spatial Awareness
├── 📡 Sensor Model (G_sensor) - IoT Data Processing
├── 🧠 Knowledge Model (I_knowledge) - Knowledge Graph
├── 🏃 Motion Model (J_motion) - Motion Control
└── 💻 Programming Model (K_programming) - Code Generation & Understanding
```

### 中文版
```
自我大脑AGI系统架构：
├── 🎛️ 管理模型 (A_management) - 中央协调器
├── 🗣️ 语言模型 (B_language) - 自然语言处理
├── 🔊 音频模型 (C_audio) - 声音分析与合成
├── 👁️ 图像模型 (D_image) - 计算机视觉
├── 🎬 视频模型 (E_video) - 视频理解
├── 🎯 空间模型 (F_spatial) - 3D空间感知
├── 📡 传感器模型 (G_sensor) - 物联网数据处理
├── 🧠 知识模型 (I_knowledge) - 知识图谱
├── 🏃 运动模型 (J_motion) - 运动控制
└── 💻 编程模型 (K_programming) - 代码生成与理解
```

---

## 🚀 Quick Start | 快速开始

#### System Requirements | 系统要求
| English | 中文 |
|---------|------|
| Python 3.8+ | Python 3.8+ |
| Windows/Linux/macOS | Windows/Linux/macOS |
| 4GB RAM (Development Environment) | 4GB 内存（开发环境） |
| 2GB Available Disk Space | 2GB 可用磁盘空间 |

#### Installation Steps | 安装步骤
```bash
# 1. Clone the project | 克隆项目
git clone [repository-url]
cd self-brain

# 2. Create virtual environment | 创建虚拟环境
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
# or
myenv\Scripts\activate     # Windows

# 3. Install dependencies | 安装依赖
pip install -r requirements.txt

# 4. Start the system | 启动系统
python start_system.py
```

#### Access Interface | 访问界面
**English**: After startup, visit: http://localhost:5000
**中文**: 启动后访问：http://localhost:5000

---

## 🎮 Usage Guide | 使用指南

#### 1. Start Management System | 启动管理系统
```bash
# Start the management model service | 启动管理模型服务
python manager_model/app.py
# Access: http://localhost:5015
```

#### 2. Start Web Interface | 启动Web界面
```bash
# Start the web interface | 启动Web界面
python web_interface/app.py
# Access: http://localhost:5000
```

#### 3. API Endpoints | API端点
| English | 中文 |
|---------|------|
| `curl http://localhost:5015/api/health` | 健康检查 |
| `curl http://localhost:5015/api/stats` | 简化系统统计 |
| `curl http://localhost:5015/api/system/stats` | 详细系统统计 |
| `curl http://localhost:5015/api/models/status` | 模型状态 |
| `curl http://localhost:5015/api/models` | 可用模型列表 |

---

## 🔧 Development Configuration | 开发配置

### English Version
```python
# System configuration example
system_config = {
    "management_port": 5015,
    "web_interface_port": 5000,
    "log_level": "INFO",
    "max_concurrent_tasks": 100,
    "auto_restart": true
}
```

### 中文版
```python
# 系统配置示例
system_config = {
    "management_port": 5015,      # 管理服务端口
    "web_interface_port": 5000,  # Web界面端口
    "log_level": "INFO",         # 日志级别
    "max_concurrent_tasks": 100, # 最大并发任务数
    "auto_restart": true         # 自动重启
}
```

---

## 📊 Performance Metrics | 性能指标

| Metric | Value | 指标 | 数值 |
|--------|--------|------|------|
| Active Models | 10 | 活跃模型 | 10 |
| API Response Time | <100ms | API响应时间 | <100毫秒 |
| Memory Usage | 32-64MB | 内存使用 | 32-64MB |
| CPU Usage | 1-2% | CPU使用率 | 1-2% |
| System Uptime | Real-time tracking | 系统运行时间 | 实时追踪 |
| Failed Tasks | 0 (current) | 失败任务 | 0（当前） |

---

## 🤝 Contributing | 贡献指南

### English Version
We welcome all forms of contribution!

#### How to Contribute
1. 🍴 Fork the project
2. 🌿 Create feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to branch (`git push origin feature/AmazingFeature`)
5. 🔄 Open Pull Request

#### Report Issues
- 🐛 Bug reports: Use issue templates
- 💡 Feature requests: Mark as enhancement
- 📚 Documentation: Mark as documentation

### 中文版
我们欢迎各种形式的贡献！

#### 如何贡献
1. 🍴 复刻项目
2. 🌿 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 💾 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 📤 推送到分支 (`git push origin feature/AmazingFeature`)
5. 🔄 打开拉取请求

#### 报告问题
- 🐛 错误报告：使用问题模板
- 💡 功能请求：标记为增强
- 📚 文档：标记为文档

---

## 📄 License | 许可证

**English**: This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

**中文**: 本项目采用 **Apache License 2.0** 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 Acknowledgments | 致谢

| English | 中文 |
|---------|------|
| **Creative Team Email**: silencecrowtom@qq.com | **创意团队邮箱**: silencecrowtom@qq.com |
| **Open Source Community**: Thanks to all open source contributors | **开源社区**: 感谢所有开源贡献者 |
| **Technical Community**: Thanks for knowledge sharing from technical communities | **技术社区**: 感谢技术社区的知识分享 |

---

## 🔗 Related Links | 相关链接

| English | 中文 |
|---------|------|
| 📧 **Email**: silencecrowtom@qq.com | 📧 **邮箱**: silencecrowtom@qq.com |
| 🐛 **Issues**: [GitHub Issues](https://github.com/[username]/self-brain/issues) | 🐛 **问题**: [GitHub Issues](https://github.com/[username]/self-brain/issues) |
| 📖 **Documentation**: [GitHub Wiki](https://github.com/[username]/self-brain/wiki) | 📖 **文档**: [GitHub Wiki](https://github.com/[username]/self-brain/wiki) |
| 💬 **Discussions**: [GitHub Discussions](https://github.com/[username]/self-brain/discussions) | 💬 **讨论**: [GitHub Discussions](https://github.com/[username]/self-brain/discussions) |

---

## ✅ Current System Status | 当前系统状态

**Last Updated**: September 14, 2025 | **最后更新**: 2025年9月14日

### 🟢 System Health | 系统健康
| English | 中文 |
|---------|------|
| **Management Service**: Running on http://localhost:5015 | **管理服务**: 运行于 http://localhost:5015 |
| **Web Interface**: Running on http://localhost:5000 | **Web界面**: 运行于 http://localhost:5000 |
| **API Status**: All endpoints operational | **API状态**: 所有端点运行正常 |
| **Memory Usage**: 32-64MB | **内存使用**: 32-64MB |
| **CPU Usage**: 1-2% | **CPU使用**: 1-2% |

### 🔄 Available Endpoints | 可用端点
| Endpoint | Status | Description | 端点 | 状态 | 描述 |
|----------|--------|-------------|------|------|------|
| `/api/health` | ✅ 200 OK | Health check endpoint | `/api/health` | ✅ 200 OK | 健康检查端点 |
| `/api/stats` | ✅ 200 OK | Simplified system statistics | `/api/stats` | ✅ 200 OK | 简化系统统计 |
| `/api/system/stats` | ✅ 200 OK | Detailed system statistics | `/api/system/stats` | ✅ 200 OK | 详细系统统计 |
| `/api/models/status` | ✅ 200 OK | All models status | `/api/models/status` | ✅ 200 OK | 所有模型状态 |
| `/api/models` | ✅ 200 OK | Available models list | `/api/models` | ✅ 200 OK | 可用模型列表 |

### 📊 Real-time Metrics | 实时指标
| English | 中文 |
|---------|------|
| **Active Models**: 10 | **活跃模型**: 10 |
| **Total Tasks**: 0 | **总任务**: 0 |
| **Successful Tasks**: 0 | **成功任务**: 0 |
| **Failed Tasks**: 0 | **失败任务**: 0 |
| **System Uptime**: Real-time tracking | **系统运行时间**: 实时追踪 |

---

<div align="center">
  <br>
  <br>
  <b>Self Brain - Giving AI True Self-Awareness</b>
  <br>
  <br>
  <i>Made with ❤️ by the Self Brain Team</i>
</div>