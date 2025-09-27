# Self Brain AGI

<div align="center">
  <img src="icons/self_brain.svg" alt="Self Brain Logo" width="200"/>
  
  <b>Self Brain - Autonomous Artificial General Intelligence System</b>
  
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.8+-lightgrey.svg)](https://python.org)
  [![AGI](https://img.shields.io/badge/Type-AGI%20System-grey.svg)]()
  
  📧 <b>Contact</b>: silencecrowtom@qq.com
</div>

---

## 🌟 Project Overview

Self Brain is an advanced autonomous AI system designed to achieve true Artificial General Intelligence (AGI) through comprehensive self-learning capabilities, model collaboration, and real-world interaction. The system integrates 11 specialized sub-models (A-K) that cover management, language processing, vision, audio analysis, spatial awareness, sensor integration, and more, all working together under a unified architecture.

All models are designed to be trained from scratch without pre-trained external models, allowing for complete control over the learning process and knowledge acquisition. The system features a clean, minimalist black, white, and gray interface with light theme preference.

---

## 🎯 Core Features

- **🔄 True从零开始训练** - All models are designed to be trained from scratch without pre-trained external models
- **🤝 Cross-Model Collaboration** - Specialized models work together to solve complex tasks
- **📊 Real-time Performance Monitoring** - Comprehensive system and model monitoring
- **🎨 Multimodal Processing** - Integrated processing of text, audio, image, and video data
- **🌐 External API Integration** - Each model can be individually connected to external API services
- **🔌 Device Control Interface** - Support for external device communication and sensor integration
- **🎥 Multi-Camera Support** - Dual camera capabilities for binocular vision and spatial perception
- **🧠 Knowledge Base Management** - Complete knowledge management with autonomous learning functionality
- **⚙️ Advanced Settings** - Fine-grained control over model parameters and system configuration
- **💻 Cross-Platform Compatibility** - Works on Windows, Linux, and macOS

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

**A_management**: Central coordinator with emotional analysis capabilities, managing all other models and human-computer interaction. It has its own emotional capacity to accept and express emotions, and communicates with users through a web-based interface.

**B_language**: Multilingual interaction model with emotional reasoning capabilities, supporting various languages and understanding emotional contexts.

**C_audio**: Audio processing model capable of speech recognition, tone analysis, voice synthesis, music recognition, noise processing, multi-band audio recognition, and various sound effects synthesis.

**D_image**: Image processing model for content recognition, editing, enhancement, size adjustment, and semantic image generation based on emotional and contextual understanding.

**E_video**: Video processing model for content recognition, editing, modification, and semantic video generation with emotional and contextual awareness.

**F_spatial**: Binocular spatial perception model for 3D space recognition, visualization modeling, positioning, distance perception, object volume recognition, motion prediction, and self-movement awareness.

**G_sensor**: Sensor data processing model for temperature, humidity, acceleration, speed, displacement, 6-axis gyroscope, pressure, air pressure, distance, infrared, taste, smoke, light, and various other sensor inputs.

**H_computer_control**: Computer system control model with multi-system compatibility for automated operations across major operating systems, enabling complete control of computer functions.

**I_motion**: Motion and actuator control model for complex control tasks with multi-port output and communication protocol compatibility, allowing control of external devices through various communication methods.

**J_knowledge**: Core knowledge expert model with comprehensive knowledge across physics, mathematics, chemistry, medicine, law, history, sociology, humanities, psychology, economics, management, mechanical engineering, electronic engineering, food engineering, chemical engineering, and other major fields of knowledge.

**K_programming**: Programming assistance model capable of autonomous code generation and system self-improvement, able to enhance the entire program based on knowledge from the knowledge base.

---

## 🚀 Quick Start

### System Requirements
- Python 3.8+
- Windows/Linux/macOS
- 8GB RAM (Minimum)
- 10GB Available Disk Space
- Web Browser (Chrome, Firefox, Edge recommended)
- Optional: External cameras, sensors, and devices for expanded functionality

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

1. **Chat Interface** - Interact with the A_management model through natural conversation in the main page
2. **Knowledge Management** - Create, import, and manage knowledge entries. Start/stop self-learning for any model.
3. **Model Training** - Train individual models or multiple models simultaneously with real training capabilities
4. **Settings** - Configure model parameters, external API connections, and system preferences
5. **Real-time Monitoring** - Track system performance and model status
6. **Device Control** - Manage connected external devices and sensors through standardized interfaces

### Knowledge Base Self-Learning
All models can learn from the knowledge base to enhance their capabilities based on their specific functions:
1. Navigate to the Knowledge Management page
2. Select one or all models from the dropdown menu
3. Click "Start Self-Learning" button
4. Monitor progress through the real-time progress bar
5. Click "Stop Self-Learning" to end the process

Each model will autonomously extract and learn knowledge relevant to its specialized domain from the knowledge base.

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
    "camera_sources": [0, 1]  # Enable multi-camera support for binocular vision
}
```

---

## 🌐 API Integration

Each model can be independently configured to use either local implementation or external API services from mainstream market providers. This allows seamless switching between local models and external AI services.

To configure external API integration:
1. Navigate to the Settings page
2. Select the model you want to configure
3. Enter the API URL, API Key, and model name using the standard external API settings
4. Click "Test Connection" to verify connectivity
5. Save the configuration to use the external model instead of the local AI model

Once configured, the external model will work seamlessly within the program, providing the same functionality as the local implementation.

---

## 🔌 External Device & Sensor Integration

Self Brain provides comprehensive support for external devices and sensors through standardized interfaces. The system includes dedicated communication interfaces for device control and sensor data acquisition.

Supported devices and communication methods:
- Serial port communication
- USB device integration
- Network-connected sensors
- Camera input for visual perception (multi-camera support)
- Sensor data acquisition (temperature, humidity, pressure, etc.)
- Actuator control interfaces

Configuration can be managed through the system settings page or directly in the sensor_config.json file, allowing flexible integration with various hardware components.

---

## 📊 Port Assignments

| Service | Port | Description |
|---------|------|-------------|
| Main Web Interface | 5000 | User interaction dashboard with clean black, white and gray light theme |
| A Management Model | 5001 | Central coordination service with emotional analysis |
| B Language Model | 5002 | Language processing service with multilingual capabilities |
| C Audio Model | 5003 | Audio processing service for speech, music, and sound analysis |
| D Image Model | 5004 | Image processing service for recognition and generation |
| E Video Model | 5005 | Video processing service for analysis and manipulation |
| F Spatial Model | 5006 | Spatial awareness service with binocular vision capabilities |
| G Sensor Model | 5007 | Sensor data processing service for various input types |
| H Computer Control | 5008 | System automation service with multi-platform support |
| I Motion Model | 5010 | Motion control service for external device management |
| J Knowledge Model | 5009 | Knowledge base service with comprehensive domain knowledge |
| K Programming Model | 5011 | Programming assistance service for code generation and system improvement |
| Manager Model API | 5015 | Advanced management API for system coordination

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
- **Self Brain Team**: Dedicated to creating truly autonomous AI systems

---

## 🔗 Related Links

- 📧 **Email**: silencecrowtom@qq.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/Sum-Outman/self-brain/issues)
- 📖 **Documentation**: [GitHub Wiki](https://github.com/Sum-Outman/self-brain/wiki)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Sum-Outman/self-brain/discussions)
- 🚀 **Project Website**: Coming Soon

---

<div align="center">
  <br>
  <b>Self Brain - Advanced Autonomous Artificial General Intelligence</b>
  <br>
  <i>Made with ❤️ by the Self Brain Team | silencecrowtom@qq.com</i>
</div>