# Self Brain AGI

<div align="center">
  <img src="icons/self_brain.svg" alt="Self Brain Logo" width="200"/>
  
  <b>Next-Generation Autonomous Artificial General Intelligence System</b>
  
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
  [![AGI](https://img.shields.io/badge/Type-AGI%20System-red.svg)]()
  
  📧 <b>Contact</b>: silencecrowtom@qq.com
</div>

---

## 🌟 Project Overview

Self Brain is a revolutionary autonomous AI system that achieves true Artificial General Intelligence through self-learning, self-optimization, and cross-domain collaboration capabilities. The system integrates 11 specialized sub-models (A-K) covering management, language, vision, audio, reasoning, and more dimensions, all working together under a unified architecture.

---

## 🎯 Core Features

- **🔄 Autonomous Training & Self-Learning** - All models can be trained from scratch and learn independently from knowledge base
- **🤝 Cross-Model Collaboration** - Models work together to solve complex tasks
- **📊 Real-time Performance Monitoring** - Comprehensive system and model monitoring
- **🎨 Multimodal Processing** - Integrated processing of text, audio, image, and video data
- **🌐 External API Integration** - Each model can be connected to external API services
- **🔌 Device Control Interface** - Support for external device communication and sensor integration
- **🎥 Multi-Camera Support** - Binocular vision capabilities for spatial perception
- **🧠 Knowledge Base Management** - Complete knowledge management and self-learning system

---

## 🏗️ System Architecture

```
Self Brain AGI System Architecture:
├── A_management - Central Management Model
├── B_language - Language Processing Model
├── C_audio - Audio Processing Model
├── D_image - Image Processing Model
├── E_video - Video Processing Model
├── F_spatial - Spatial Awareness Model
├── G_sensor - Sensor Data Processing Model
├── H_computer_control - Computer Control Model
├── I_motion - Motion & Actuator Control Model
├── J_knowledge - Knowledge Base Expert Model
└── K_programming - Programming Assistant Model
```

### Detailed Model Descriptions:

**A_management**: Central coordinator with emotional analysis capabilities, managing all other models and human-computer interaction.

**B_language**: Multilingual interaction model with emotional reasoning capabilities.

**C_audio**: Audio processing model capable of speech recognition, tone analysis, voice synthesis, music recognition, and noise processing.

**D_image**: Image processing model for content recognition, editing, enhancement, and semantic image generation.

**E_video**: Video processing model for content recognition, editing, modification, and semantic video generation.

**F_spatial**: Binocular spatial perception model for 3D space recognition, modeling, positioning, distance perception, and motion prediction.

**G_sensor**: Sensor data processing model for temperature, humidity, acceleration, gyroscope, pressure, light, and various other sensors.

**H_computer_control**: Computer system control model with multi-system compatibility for automated operations.

**I_motion**: Motion and actuator control model for complex control tasks with multi-port output and communication protocol compatibility.

**J_knowledge**: Core knowledge expert model with comprehensive knowledge across physics, mathematics, chemistry, medicine, law, history, psychology, engineering, and more fields.

**K_programming**: Programming assistance model capable of autonomous code generation and system self-improvement.

---

## 🚀 Quick Start

### System Requirements
- Python 3.8+
- Windows/Linux/macOS
- 8GB RAM (Minimum)
- 10GB Available Disk Space
- Web Browser (Chrome, Firefox, Edge recommended)
- Optional: External cameras, sensors, and devices

### Installation Steps
```bash
# 1. Clone the repository
git clone https://github.com/Sum-Outman/self-brain.git
cd self-brain

# 2. Create virtual environment
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
# or
myenv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the system
python start_system.py
```

### Access Interface
After startup, visit: http://localhost:5000

---

## 🎮 Usage Guide

### Starting Services
```bash
# Start the complete system (recommended)
python start_system.py

# or start individual components
# Start management system
python a_manager_standalone.py

# Start web interface only
python web_interface/app.py
```

### Key Functional Areas

1. **Chat Interface** - Interact with the A_management model through natural conversation
2. **Knowledge Management** - Create, import, and manage knowledge entries. Start/stop self-learning for any model.
3. **Model Training** - Train individual models or multiple models simultaneously
4. **Settings** - Configure model parameters, external API connections, and system preferences
5. **Real-time Monitoring** - Track system performance and model status
6. **Device Control** - Manage connected external devices and sensors

### Knowledge Base Self-Learning
To start self-learning for models on the knowledge base:
1. Navigate to the Knowledge Management page
2. Select one or all models from the dropdown menu
3. Click "Start Self-Learning" button
4. Monitor progress through the real-time progress bar
5. Click "Stop Self-Learning" to end the process

---

## 🔧 Development Configuration

Key configuration files:
- `config/system_config.yaml` - Main system configuration
- `config/training_config.json` - Model training parameters
- `config/model_registry.json` - Registered models information
- `config/sensor_config.json` - Sensor configuration

```python
# Example system configuration snippet
{
    "management_port": 5001,
    "web_interface_port": 5000,
    "log_level": "INFO",
    "max_concurrent_tasks": 100,
    "auto_restart": true,
    "camera_sources": [0, 1]  # Enable multi-camera support
}
```

---

## 🌐 API Integration

Each model can be configured to use either local implementation or external API services. To configure external API integration:
1. Navigate to the Settings page
2. Select the model you want to configure
3. Enter the API URL, API Key, and model name
4. Click "Test Connection" to verify connectivity
5. Save the configuration to use the external model

---

## 🔌 External Device & Sensor Integration

Self Brain supports a wide range of external devices and sensors through standardized interfaces:
- Serial port communication
- USB device integration
- Network-connected sensors
- Camera input for visual perception

Configuration can be managed through the system settings page or directly in the sensor_config.json file.

---

## 📊 Port Assignments

| Service | Port | Description |
|---------|------|-------------|
| Main Web Interface | 5000 | User interaction dashboard |
| A Management Model | 5001 | Central coordination service |
| B Language Model | 5002 | Language processing service |
| C Audio Model | 5003 | Audio processing service |
| D Image Model | 5004 | Image processing service |
| E Video Model | 5005 | Video processing service |
| F Spatial Model | 5006 | Spatial awareness service |
| G Sensor Model | 5007 | Sensor data processing service |
| H Computer Control | 5008 | System automation service |
| I Motion Model | 5010 | Motion control service |
| J Knowledge Model | 5009 | Knowledge base service |
| K Programming Model | 5011 | Programming assistance service |
| Manager Model API | 5015 | Advanced management API |

---

## 🤝 Contributing

We welcome all forms of contribution to the Self Brain project!

### How to Contribute
1. 🍴 Fork the project
2. 🌿 Create feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to branch (`git push origin feature/AmazingFeature`)
5. 🔄 Open Pull Request

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Creative Team Email**: silencecrowtom@qq.com
- **Open Source Community**: Thanks to all open source contributors
- **Technical Community**: Thanks for knowledge sharing from technical communities

---

## 🔗 Related Links

- 📧 **Email**: silencecrowtom@qq.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/Sum-Outman/self-brain/issues)
- 📖 **Documentation**: [GitHub Wiki](https://github.com/Sum-Outman/self-brain/wiki)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Sum-Outman/self-brain/discussions)

---

<div align="center">
  <br>
  <b>Self Brain - Giving AI True Self-Awareness</b>
  <br>
  <i>Made with ❤️ by the Self Brain Team</i>
</div>