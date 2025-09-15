# 🐳 Self Brain AGI Docker Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying Self Brain AGI using Docker containers, ensuring consistent and scalable deployment across different environments.

## 🚀 Quick Start (One Command)

```bash
# Production deployment
./docker-deploy.sh prod

# Development deployment  
./docker-deploy.sh dev
```

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM (8GB+ recommended for production)
- 2GB+ disk space

## 🏗️ Architecture

### Services Overview

| Service | Port | Description |
|---------|------|-------------|
| nginx | 80 | Reverse proxy & load balancer |
| manager | 5000 | Central management service |
| a_management | 5001 | A management model |
| b_language | 5002 | Language processing |
| c_audio | 5003 | Audio analysis |
| d_image | 5004 | Image processing |
| e_video | 5005 | Video analysis |
| f_spatial | 5006 | Spatial awareness |
| g_sensor | 5007 | Sensor data |
| h_computer_control | 5008 | System control |
| i_knowledge | 5009 | Knowledge base |
| j_motion | 5010 | Motion control |
| k_programming | 5011 | Code generation |
| prometheus | 9090 | Metrics collection |
| grafana | 3000 | Monitoring dashboard |

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```bash
# Flask settings
FLASK_ENV=production
FLASK_DEBUG=0
SECRET_KEY=your-secure-secret-key

# Database
DATABASE_URL=sqlite:///knowledge_base.db

# Ports
NGINX_PORT=80
MANAGER_PORT=5000
A_MANAGEMENT_PORT=5001
B_LANGUAGE_PORT=5002
C_AUDIO_PORT=5003
D_IMAGE_PORT=5004
E_VIDEO_PORT=5005
F_SPATIAL_PORT=5006
G_SENSOR_PORT=5007
H_CONTROL_PORT=5008
I_KNOWLEDGE_PORT=5009
J_MOTION_PORT=5010
K_PROGRAMMING_PORT=5011

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

## 🏃‍♂️ Deployment Options

### 1. Production Deployment

```bash
# Using deployment script
./docker-deploy.sh prod

# Manual deployment
docker-compose -f docker-compose.prod.yml up -d
```

### 2. Development Deployment

```bash
# Using deployment script
./docker-deploy.sh dev

# Manual deployment
docker-compose -f docker-compose.dev.yml up -d
```

### 3. Custom Deployment

```bash
# Build and start specific services
docker-compose up -d manager nginx

# Scale specific services
docker-compose up -d --scale b_language=3
```

## 📊 Monitoring

### Access Monitoring Dashboard

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### Health Checks

```bash
# Check all services
./docker-deploy.sh health

# Individual service health
curl http://localhost:5000/api/health
curl http://localhost:5001/api/health
```

## 🔍 Logs & Debugging

### View Logs

```bash
# All services
./docker-deploy.sh logs

# Specific service
docker-compose logs -f manager
docker-compose logs -f nginx
```

### Debug Mode

```bash
# Development with debug logs
docker-compose -f docker-compose.dev.yml up --build
```

## 🛡️ Security

### Security Features

- **Non-root containers**: All services run as non-root user
- **Network isolation**: Custom Docker networks
- **Volume permissions**: Proper file ownership
- **Security headers**: Nginx security configuration
- **Rate limiting**: API protection via Nginx

### SSL/TLS Setup

```bash
# Enable HTTPS (production)
# Edit nginx.conf to uncomment SSL configuration
# Mount SSL certificates to /etc/nginx/ssl/
```

## 🔄 Updates & Maintenance

### Update System

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
./docker-deploy.sh prod
```

### Backup & Restore

```bash
# Backup data volumes
docker run --rm -v selfbrain_knowledge_data:/data -v $(pwd):/backup alpine tar czf /backup/knowledge_backup.tar.gz /data

# Restore data volumes
docker run --rm -v selfbrain_knowledge_data:/data -v $(pwd):/backup alpine sh -c "cd /data && tar xzf /backup/knowledge_backup.tar.gz --strip 1"
```

## 🧪 Testing

### Automated Testing

```bash
# Run system tests
docker-compose exec manager python system_validation.py

# API testing
curl -X GET http://localhost:5000/api/models
curl -X POST http://localhost:5000/api/process -H "Content-Type: application/json" -d '{"input": "test"}'
```

### Load Testing

```bash
# Install artillery for load testing
npm install -g artillery

# Run load test
artillery quick --count 50 --num 10 http://localhost:5000/api/health
```

## 🚨 Troubleshooting

### Common Issues

1. **Port conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :5000
   
   # Change ports in .env file
   MANAGER_PORT=5000
   ```

2. **Permission issues**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER .
   chmod +x docker-deploy.sh
   ```

3. **Memory issues**
   ```bash
   # Check Docker memory
   docker system df
   docker system prune -a
   ```

4. **Service startup failures**
   ```bash
   # Check service logs
   docker-compose logs [service-name]
   
   # Restart specific service
   docker-compose restart [service-name]
   ```

### Performance Tuning

```bash
# Increase Docker memory limits
# Edit Docker daemon settings:
{
  "default-ulimits": {
    "memlock": {
      "Hard": -1,
      "Name": "memlock",
      "Soft": -1
    }
  },
  "default-memory": 2147483648,
  "default-memory-swap": 2147483648
}
```

## 📱 Docker Commands Reference

### Basic Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart services
docker-compose restart

# View status
docker-compose ps

# Scale services
docker-compose up -d --scale b_language=3
```

### Advanced Commands

```bash
# Build without cache
docker-compose build --no-cache

# Execute commands in running container
docker-compose exec manager python manage.py migrate

# Copy files to/from containers
docker cp local_file.txt container_name:/app/
```

## 🌐 Cloud Deployment

### AWS ECS

```bash
# Install ECS CLI
curl -Lo ecs-cli https://amazon-ecs-cli.s3.amazonaws.com/ecs-cli-linux-amd64-latest
chmod +x ecs-cli

# Configure ECS
ecs-cli configure --cluster selfbrain --region us-west-2
```

### Kubernetes

```bash
# Convert docker-compose to Kubernetes
kompose convert -f docker-compose.prod.yml

# Deploy to Kubernetes
kubectl apply -f .
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Sum-Outman/self-brain/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Sum-Outman/self-brain/discussions)
- **Email**: silencecrowtom@qq.com

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Docker Hub
        run: |
          docker-compose -f docker-compose.prod.yml build
          docker-compose -f docker-compose.prod.yml push
```

---

<p align="center">
  <strong>⭐ Star this repository if you find it helpful! ⭐</strong>
</p>