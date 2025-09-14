# Self Brain AGI System - Port Allocation Guide

## System Port Mapping (Updated to Avoid Conflicts)

| Service | Port | Status | Description |
|---------|------|---------|-------------|
| **Main Web Interface** | 5000 | ✅ Active | Primary web interface |
| **A Management Model** | 5001 | ✅ Active | Core management system |
| **B Language Model** | 5002 | ✅ Active | Natural language processing |
| **C Audio Model** | 5003 | ✅ Active | Audio processing |
| **D Image Model** | 5004 | ✅ Active | Computer vision |
| **E Video Model** | 5005 | ✅ Active | Video analysis |
| **F Spatial Model** | 5006 | ✅ Active | 3D spatial processing |
| **G Sensor Model** | 5007 | ✅ Active | IoT sensor integration |
| **H Computer Control** | 5008 | ✅ Active | System control |
| **I Knowledge Model** | 5009 | ✅ Active | Knowledge base |
| **J Motion Model** | 5010 | ✅ Active | Motion control |
| **K Programming Model** | 5011 | ✅ Active | Code generation |
| **Training Manager** | 5012 | ✅ Active | Training coordination |
| **Quantum Integration** | 5013 | ✅ Active | Quantum computing interface |
| **Standalone A Manager** | 5014 | ✅ New | Independent A management |
| **Manager Model API** | 5015 | ✅ New | Manager model standalone |
| **Development Server** | 8080 | ✅ Reserved | Development/debugging |
| **Backup Services** | 8000-8010 | ✅ Reserved | Backup and testing |

## Updated Configuration Files

### Primary Services
- **Web Interface**: 5000 (unchanged)
- **A Management Model**: 5001 (unchanged)
- **B Language Model**: 5002 (unchanged)
- **C Audio Model**: 5003 (unchanged)
- **D Image Model**: 5004 (unchanged)
- **E Video Model**: 5005 (unchanged)
- **F Spatial Model**: 5006 (unchanged)
- **G Sensor Model**: 5007 (unchanged)
- **H Computer Control**: 5008 (unchanged)
- **I Knowledge Model**: 5009 (unchanged)
- **J Motion Model**: 5010 (unchanged)
- **K Programming Model**: 5011 (moved from 5010)

### New/Updated Services
- **Training Manager**: 5012 (new)
- **Quantum Integration**: 5013 (moved from 5011)
- **Standalone A Manager**: 5014 (new)
- **Manager Model API**: 5015 (new)

## Usage Instructions

### Start All Services
```bash
# Start the complete system
python start_system.py

# Start standalone A Manager (new port 5014)
python a_manager_standalone.py

# Start manager model API (new port 5015)
cd manager_model
python app.py
```

### Test Individual Services
```bash
# Test A Management Model
curl http://localhost:5001/api/health

# Test Standalone A Manager
curl http://localhost:5014/api/health

# Test Manager Model API
curl http://localhost:5015/api/health
```

### Environment Variables
```bash
# Override default ports
export PORT_A_MANAGER=5001
export PORT_STANDALONE=5014
export PORT_MANAGER_API=5015
```

## Port Conflict Resolution

Previous conflicts resolved:
- ✅ K Programming Model: 5010 → 5011
- ✅ Quantum Integration: 5011 → 5013
- ✅ Standalone A Manager: 5003 → 5014
- ✅ Manager Model API: 5001 → 5015
- ✅ All training endpoints updated to new port ranges