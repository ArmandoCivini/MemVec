# MemVec API

RESTful API layer for MemVec vector database operations.

## Overview

The MemVec API provides a web interface to interact with the MemVec vector database system. It exposes endpoints for storing vectors, querying similar vectors, and managing the system.

## Features

- **Upload Files**: Process documents and convert them to vectors
- **Query Vectors**: Search for similar content using text queries
- **File Management**: List and manage processed files
- **RESTful**: Standard HTTP endpoints with JSON payloads
- **Async**: Asynchronous operations for better performance

## Installation

Install the API dependencies:

```bash
pip install fastapi uvicorn pydantic python-multipart
```

## Running the API

Start the development server:

```bash
python api/server.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /
```

### Upload File
```
POST /upload
Content-Type: multipart/form-data

file: [file data]
metadata: {"optional": "metadata"} (optional JSON string)
```

### List Files
```
GET /files
```

### Query Vectors
```
POST /query
{
  "query_text": "search text",
  "k": 5,
  "threshold": 0.8
}
```

### Delete File
```
DELETE /files/{filename}
```

### System Statistics
```
GET /stats
```

## TODO

- **File Deletion**: Complete implementation of file deletion logic
  - Remove vectors from FAISS index
  - Remove vector chunks from S3 storage
  - Clean up all related cache entries
  - Handle cascading deletions properly

## Usage Example

```python
from api.client import MemVecClient

client = MemVecClient()

# Upload a file
result = client.upload_file(
    "document.txt",
    metadata={"source": "example"}
)

# Query vectors
results = client.query_vectors("artificial intelligence", k=5)

# List processed files
files = client.list_files()
```

## Architecture

- **FastAPI**: Web framework for building the API
- **Pydantic**: Data validation and serialization
- **Service Layer**: Business logic orchestration
- **MemVec Core**: Integration with the core MemVec system

## Development

The API is structured with:

- `main.py`: FastAPI application and route definitions
- `models.py`: Pydantic models for requests/responses
- `service.py`: Business logic and MemVec integration
- `client.py`: Example client for testing
- `server.py`: Development server startup script
