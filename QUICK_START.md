# Self Brain AGI - Quick Start Guide

## 30-Second Deployment

```bash
git clone https://github.com/Sum-Outman/self-brain.git
cd self-brain
./docker-deploy.sh prod
```

## Service URLs

| Service | URL | Status |
|---------|-----|--------|
| Main Interface | http://localhost:5000 | ✅ |
| Management | http://localhost:5001 | ✅ |
| Language | http://localhost:5002 | ✅ |
| Audio | http://localhost:5003 | ✅ |
| Image | http://localhost:5004 | ✅ |
| Video | http://localhost:5005 | ✅ |
| Spatial | http://localhost:5006 | ✅ |
| Sensor | http://localhost:5007 | ✅ |
| Control | http://localhost:5008 | ✅ |
| Knowledge | http://localhost:5009 | ✅ |
| Motion | http://localhost:5010 | ✅ |
| Programming | http://localhost:5011 | ✅ |
| Monitoring | http://localhost:3000 | ✅ |

## Health Check

```bash
./docker-deploy.sh health
```

## Stop All Services

```bash
./docker-deploy.sh stop
```