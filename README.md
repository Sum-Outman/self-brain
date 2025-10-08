# Self Brain - Advanced General Intelligence System

## Overview
Self Brain is a comprehensive Artificial General Intelligence (AGI) management system designed to coordinate multiple AI models for language processing, computer vision, audio processing, spatial awareness, and device control. The system features a clean, minimalist black-and-white web interface and supports local training as well as integration with external AI APIs.

## System Architecture

The system consists of 11 core models working together to provide a unified AGI experience:

1. **A - Management Model**: Main coordinator that handles human interaction and task routing
2. **B - Language Model**: Natural language processing and generation
3. **C - Audio Processing Model**: Speech recognition and synthesis
4. **D - Image Processing Model**: Visual content recognition and generation
5. **E - Video Processing Model**: Video content analysis and manipulation
6. **F - Spatial Perception Model**: 3D space understanding and object tracking
7. **G - Sensor Model**: Environmental data collection and processing
8. **H - Computer Control Model**: System automation and control functions
9. **I - Motion Control Model**: Actuator control and movement planning
10. **J - Knowledge Base Model**: Comprehensive knowledge repository and expert system
11. **K - Programming Model**: Code generation and system self-improvement

## Port Configuration

The system uses the following port assignments for its components:

| Service | Port | Description |
|---------|------|------------ |
| Main Web Interface | 8080 | Primary user interface |
| A Management Model | 5000 | Core management system |
| B Language Model | 5001 | Natural language processing |
| C Audio Model | 5002 | Audio processing |
| D Image Model | 5003 | Computer vision |
| E Video Model | 5004 | Video analysis |
| F Spatial Model | 5005 | 3D spatial processing |
| G Sensor Model | 5006 | IoT sensor integration |
| H Computer Control | 5007 | System control |
| I Knowledge Model | 5008 | Knowledge base |
| J Motion Model | 5009 | Motion control |
| K Programming Model | 5010 | Code generation |
| AGI Core | 5014 | Advanced AGI processing |
| Manager Model API | 5015 | Manager model standalone API |

## Installation Requirements

To run the Self Brain system, you'll need:

- Python 3.6 or higher
- Required packages (see requirements.txt)
- Web browser for the user interface
- Optional: External sensors and devices for enhanced functionality

## Installation Steps

### Quick Deployment (Recommended)

For Windows systems, use the provided deployment scripts:

**Option 1: PowerShell (Recommended)**
```powershell
.\deploy_system.ps1
```

**Option 2: Batch File**
```cmd
deploy_system.bat
```

These scripts will automatically:
- Create and activate a Python virtual environment
- Install all required dependencies
- Initialize system configuration
- Start the web interface
- Provide system health checks

### Manual Installation

1. Clone or download the repository
2. Navigate to the project directory: `cd d:\shiyan`
3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the system startup script:
   ```
   python web_interface\app.py
   ```
5. Access the web interface at http://localhost:8080

## Key Features

### System Architecture
- **A Management Model**: Main coordinator that handles human interaction and task routing
- **B Language Model**: Natural language processing and generation with emotional reasoning
- **C Audio Model**: Speech recognition, tone analysis, and audio synthesis
- **D Vision Model**: Image recognition, manipulation, and generation
- **E Video Model**: Video content analysis, editing, and generation
- **F Spatial Model**: 3D space understanding, modeling, and positioning
- **G Sensor Model**: Environmental data collection and processing
- **H Computer Control Model**: System automation and control functions
- **I Motion Control Model**: Actuator control and movement planning
- **J Knowledge Base Model**: Comprehensive knowledge repository and expert system
- **K Programming Model**: Code generation and system self-improvement

### Web Interface
- Clean black-and-white minimalist design
- Real-time English chat interface with the management model
- Multi-camera support for stereo vision applications
- Model training controls (individual and joint training)
- Device communication management
- System status monitoring

### Training Capabilities
- Knowledge learning, reinforcement learning, supervised, and unsupervised training modes
- External training data upload
- Real-time training status monitoring
- Integration with the knowledge model for self-improvement
- **Note**: Model pre-training system is currently being optimized for all 11 core models

## Usage Guide

1. **Chat Interface**: Communicate with the AI through text in English or voice
2. **Knowledge Management**: View, edit, and expand the knowledge base
3. **Training Control**: Train models with custom data and monitor progress
4. **System Settings**: Configure models, APIs, and hardware integrations
5. **Camera Management**: Set up and calibrate multiple cameras for visual perception
6. **Device Communication**: Connect and control external hardware devices

## Project Structure
- **web_interface/**: Web application frontend and backend
  - **templates/**: HTML templates for the web interface (using index.html as main page)
  - **static/**: Static assets (CSS, JavaScript)
  - **backend/**: Server-side API implementations
- **training_manager/**: Model training logic
- **unified_device_communication/**: Device integration modules
- **camera_manager.py**: Camera control and processing
- **app.py**: Main application entry point
- **deploy_system.bat**: Windows batch deployment script
- **deploy_system.ps1**: PowerShell deployment script
- **requirements.txt**: Project dependencies

## Project Information

- **Name**: Self Brain
- **Development Team**: Silence Crow Team
- **Contact**: silencecrowtom@qq.com
- **Version**: 1.0.0
- **Current Status**: System running on port 8080 with full web interface functionality
- **Deployment**: Enhanced deployment scripts available for automated setup

## License

This project is proprietary software developed by the Silence Crow Team.
