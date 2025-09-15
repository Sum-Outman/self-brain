# Self Brain AGI - Docker Deployment Guide

This guide provides comprehensive instructions for deploying the Self Brain AGI system using Docker containers.

## Quick Start

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- 8GB+ RAM recommended
- 10GB+ free disk space

### One-Command Deployment
```bash
# Development environment
./docker-deploy.sh dev

# Production environment
./docker-deploy.sh prod
```

## Available Services

| Service | Port | Description |
|---------|------|-------------|
| Main Interface | 5000 | Primary web interface |
| D Image | 5004 | Image processing service |
| E Video | 5005 | Video processing service |
| F Spatial | 5006 | Spatial positioning service |
| G Sensor | 5007 | Sensor data service |
| H Control | 5008 | Computer control service |
| I Knowledge | 5009 | Knowledge base service |
| J Motion | 5010 | Motion control service |
| K Programming | 5011 | Programming service |

## Deployment Options

### 1. Development Environment
```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Or use the deployment script
./docker-deploy.sh dev
```

Features:
- Hot-reload enabled
- Volume mounting for live code changes
- Debug mode enabled
- All ports exposed

### 2. Production Environment
```bash
# Start production environment
docker-compose -f docker-compose.prod.yml up -d

# Or use the deployment script
./docker-deploy.sh prod
```

Features:\- Gunicorn WSGI server
- Health checks enabled
- Restart policies configured
- Optimized for performance

### 3. Basic Environment
```bash
# Start basic environment
docker-compose up -d
```

## Management Commands

### Using the Deployment Script
```bash
# View all commands
./docker-deploy.sh

# Stop all services
./docker-deploy.sh stop

# View logs
./docker-deploy.sh logs
./docker-deploy.sh logs [service_name]

# Check service health
./docker-deploy.sh health

# Clean up Docker resources
./docker-deploy.sh clean
```

### Manual Commands
```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Remove volumes
docker-compose down -v
```

## Configuration

### Environment Variables
Create a `.env` file in the project root:
```bash
# Flask settings
FLASK_ENV=production
FLASK_DEBUG=0

# Service ports
PORT=5000
D_IMAGE_PORT=5004
E_VIDEO_PORT=5005
F_SPATIAL_PORT=5006
G_SENSOR_PORT=5007
H_CONTROL_PORT=5008
I_KNOWLEDGE_PORT=5009
J_MOTION_PORT=5010
K_PROGRAMMING_PORT=5011

# Security
SECRET_KEY=your-secret-key-here
```

### Volume Mounting
The following directories are mounted as volumes:
- `./data` - Application data
- `./logs` - Log files
- `./models` - ML models
- `./uploads` - File uploads
- `./static` - Static files

## Health Checks

All services provide health check endpoints:
```bash
# Check individual service health
curl http://localhost:5000/health
curl http://localhost:5004/health
# ... etc for all ports
```

## Monitoring

### Prometheus (Production)
- URL: http://localhost:9090
- Metrics collection configured

### Grafana (Production)
- URL: http://localhost:3000
- Default credentials: admin/admin
- Pre-configured dashboards

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Check port usage
   netstat -tulpn | grep :5000
   
   # Or use different ports
   docker-compose -f docker-compose.dev.yml up -d --scale manager=0
   ```

2. **Permission denied on Windows**
   ```bash
   # Run as administrator or use WSL2
   docker-compose up -d
   ```

3. **Out of memory errors**
   ```bash
   # Increase Docker memory limit
   # Docker Desktop > Settings > Resources > Memory > 8GB+
   ```

4. **Build failures**
   ```bash
   # Clear build cache
   docker-compose build --no-cache
   
   # Check system requirements
   docker system df
   ```

### Log Analysis
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f manager
docker-compose logs -f d_image

# Search logs
docker-compose logs | grep ERROR
docker-compose logs | grep WARNING
```

### Performance Tuning

#### Production Optimizations
1. **Resource limits** (docker-compose.prod.yml):
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2.0'
         memory: 4G
       reservations:
         cpus: '1.0'
         memory: 2G
   ```

2. **Gunicorn workers**:
   ```bash
   # Calculate optimal workers
   workers = (CPU cores * 2) + 1
   ```

3. **Database connections**:
   ```yaml
   environment:
     - SQLALCHEMY_POOL_SIZE=20
     - SQLALCHEMY_MAX_OVERFLOW=30
   ```

## Security Considerations

### Production Deployment
1. **Use HTTPS**
   - Configure SSL certificates
   - Update nginx.conf for HTTPS

2. **Network security**
   - Use Docker networks
   - Restrict port exposure

3. **Secrets management**
   ```bash
   # Use Docker secrets
   docker secret create db_password mypassword.txt
   ```

4. **Regular updates**
   ```bash
   # Update base images
   docker-compose pull
   docker-compose up -d
   ```

## Backup and Recovery

### Data Backup
```bash
# Backup data volume
docker run --rm -v selfbrain_data:/data -v $(pwd):/backup alpine tar czf /backup/data-backup.tar.gz /data

# Backup logs
docker run --rm -v selfbrain_logs:/logs -v $(pwd):/backup alpine tar czf /backup/logs-backup.tar.gz /logs
```

### Recovery
```bash
# Restore data
docker run --rm -v selfbrain_data:/data -v $(pwd):/backup alpine sh -c "cd / && tar xzf /backup/data-backup.tar.gz"
```

## Support

For issues and questions:
1. Check service logs: `./docker-deploy.sh logs`
2. Verify health checks: `./docker-deploy.sh health`
3. Review configuration files
4. Check system resources

## Next Steps

1. **SSL Configuration**: Set up HTTPS with Let's Encrypt
2. **Load Balancing**: Configure multiple instances
3. **Auto-scaling**: Implement Kubernetes deployment
4. **CI/CD**: Add GitHub Actions for automated deployment