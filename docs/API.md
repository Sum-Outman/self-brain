# API Documentation

## AI Management System API Reference

### Base URL
```
http://localhost:5000/api
```

### Authentication
All endpoints require API key in header:
```
Authorization: Bearer YOUR_API_KEY
```

---

## System Status

### GET /system/status
Returns overall system health and status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "models": {
    "total": 11,
    "active": 9,
    "training": 2
  },
  "services": {
    "web": "running",
    "manager": "running",
    "training": "active"
  }
}
```

---

## Model Management

### GET /models/status
Returns status of all available models.

**Response:**
```json
{
  "models": [
    {
      "id": "B_language",
      "name": "Language Processing",
      "status": "ready",
      "version": "1.0.0",
      "last_trained": "2024-01-01T10:00:00Z"
    }
  ]
}
```

### POST /models/train
Start training for specified model.

**Request:**
```json
{
  "model_id": "B_language",
  "training_data": "path/to/data.json",
  "parameters": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "joint_training": false
}
```

**Response:**
```json
{
  "training_id": "uuid-123",
  "status": "started",
  "estimated_time": "2h 30m"
}
```

### GET /models/{model_id}/status
Get specific model status.

---

## Data Upload

### POST /upload/training-data
Upload training data for models.

**Form Data:**
- `files`: Array of files
- `model`: Target model ID
- `joint_training`: boolean (optional)
- `external_api`: boolean (optional)

**Response:**
```json
{
  "upload_id": "uuid-456",
  "files_processed": 5,
  "total_size": "125MB",
  "status": "processing"
}
```

### GET /upload/status/{upload_id}
Check upload progress.

---

## Knowledge Base

### GET /knowledge/search
Search knowledge base.

**Query Parameters:**
- `query`: Search string
- `limit`: Number of results (default: 10)
- `type`: Filter by type

**Response:**
```json
{
  "results": [
    {
      "id": "knowledge-123",
      "title": "Machine Learning Basics",
      "content": "...",
      "relevance": 0.95,
      "type": "documentation"
    }
  ]
}
```

### POST /knowledge/learn
Add new knowledge.

**Request:**
```json
{
  "title": "New Concept",
  "content": "Detailed content...",
  "tags": ["ai", "ml"],
  "source": "user_input"
}
```

---

## Training Control

### GET /training/status
Get active training jobs.

### POST /training/stop/{training_id}
Stop specific training job.

### GET /training/history
Get training history.

---

## Real-time Communication

### WebSocket /ws/training-updates
Subscribe to training updates.

**Message Format:**
```json
{
  "type": "training_update",
  "training_id": "uuid-123",
  "progress": 45.5,
  "status": "epoch_3_of_10",
  "metrics": {
    "loss": 0.123,
    "accuracy": 0.95
  }
}
```

---

## Error Handling

### Standard Error Response
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "model_id",
      "issue": "Model not found"
    }
  }
}
```

### Error Codes
- `400` Bad Request
- `404` Not Found
- `429` Rate Limited
- `500` Internal Server Error

---

## Rate Limiting
- 100 requests per minute per API key
- Training endpoints: 5 concurrent jobs

## Pagination
List endpoints support pagination:
```
GET /models/status?page=1&limit=20
```