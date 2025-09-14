# Installation Guide

## AI Management System Setup

### System Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space
- **GPU**: Optional (CUDA support for training acceleration)

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-org/ai-management-system.git
cd ai-management-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start Services
```bash
# Start manager service
python manager_model/app.py

# In new terminal, start web interface
python web_interface/app.py
```

### 5. Access System
- **Web Interface**: http://localhost:5000
- **API**: http://localhost:5000/api
- **Manager**: http://localhost:5015

---

## Detailed Installation

### Python Environment Setup

#### Windows
```powershell
# Install Python 3.8+ from python.org
# Create virtual environment
python -m venv ai_env
ai_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### macOS
```bash
# Install via Homebrew
brew install python@3.8

# Create environment
python3 -m venv ai_env
source ai_env/bin/activate

pip install -r requirements.txt
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.8 python3-pip

python3 -m venv ai_env
source ai_env/bin/activate

pip install -r requirements.txt
```

### GPU Support (Optional)

#### CUDA Installation
```bash
# For NVIDIA GPUs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### AMD GPU
```bash
# ROCm support (Linux only)
pip install torch-rocm
```

---

## Configuration

### 1. Environment Variables
Create `.env` file:
```bash
# API Configuration
API_KEY=your-secret-key
DEBUG=true

# Database
DATABASE_URL=sqlite:///ai_system.db

# Model Paths
MODEL_STORAGE_PATH=./models
TRAINING_DATA_PATH=./training_data

# External APIs
OPENAI_API_KEY=your-openai-key
HUGGINGFACE_TOKEN=your-hf-token
```

### 2. Model Configuration
Edit `config/model_registry.json`:
```json
{
  "models": {
    "B_language": {
      "enabled": true,
      "gpu": true,
      "max_memory": "4GB"
    }
  }
}
```

---

## Service Management

### Development Mode
```bash
# Start all services
python start_system.py

# Or individually
python manager_model/app.py --port 5015
python web_interface/app.py --port 5000
```

### Production Mode
```bash
# Using Docker
docker-compose up -d

# Using systemd (Linux)
sudo systemctl start ai-manager
sudo systemctl start ai-web
```

---

## Verification

### 1. Health Check
```bash
curl http://localhost:5000/api/system/status
```

### 2. Model Status
```bash
curl http://localhost:5000/api/models/status
```

### 3. Web Interface
Open browser: http://localhost:5000

---

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check port usage
netstat -ano | findstr :5000  # Windows
lsof -i :5000                # Mac/Linux

# Change port
python web_interface/app.py --port 5001
```

#### Memory Issues
```bash
# Reduce model memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Use CPU only
export CUDA_VISIBLE_DEVICES=""
```

#### Import Errors
```bash
# Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

### Log Files
- Web Interface: `logs/web_interface.log`
- Manager: `logs/manager.log`
- Training: `logs/training.log`

---

## Docker Installation

### Using Docker Compose
```bash
# Build and run
docker-compose up --build

# Background mode
docker-compose up -d
```

### Manual Docker
```bash
# Build images
docker build -t ai-manager .
docker build -t ai-web -f web_interface/Dockerfile .

# Run containers
docker run -p 5015:5015 ai-manager
docker run -p 5000:5000 ai-web
```

---

## Security Notes

### Production Checklist
- [ ] Change default API keys
- [ ] Use HTTPS in production
- [ ] Configure firewall rules
- [ ] Set up log rotation
- [ ] Enable rate limiting
- [ ] Use secure database

### Network Security
```bash
# Firewall rules (Ubuntu)
sudo ufw allow 5000/tcp
sudo ufw allow 5015/tcp
sudo ufw enable
```

---

## Support

### Getting Help
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@ai-system.com

### System Requirements Check
```bash
python scripts/check_requirements.py
```