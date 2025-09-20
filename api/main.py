"""
Main FastAPI application for MemVec API.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .models import QueryRequest, FileUploadResponse, QueryResponse
from .service import MemVecService

app = FastAPI(
    title="MemVec API",
    description="RESTful API for MemVec vector database operations",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MemVec service as server state
memvec_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    global memvec_service
    memvec_service = MemVecService()


@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown."""
    global memvec_service
    if memvec_service:
        # Could add cleanup logic here if needed
        pass


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "MemVec API is running", "version": "0.1.0"}


@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """
    Upload and process a file in MemVec system.
    
    Args:
        file: The file to upload and process
        metadata: Optional JSON string with file metadata
        
    Returns:
        Response with processing status and vector IDs
    """
    try:
        # Parse metadata if provided
        file_metadata = {}
        if metadata:
            import json
            file_metadata = json.loads(metadata)
        
        result = await memvec_service.process_file(
            file=file,
            metadata=file_metadata
        )
        return FileUploadResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_vectors(request: QueryRequest):
    """
    Query vectors from MemVec system.
    
    Args:
        request: Query request containing search text and parameters
        
    Returns:
        Response with matching vectors and metadata
    """
    try:
        result = await memvec_service.query_vectors(
            query_text=request.query_text,
            k=request.k,
            threshold=request.threshold
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """
    Delete a processed file and its vectors.
    
    Args:
        filename: The name of the file to delete
        
    Returns:
        Deletion status
    """
    try:
        result = await memvec_service.delete_file(filename)
        return {"success": result, "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
async def list_files():
    """
    List all processed files.
    
    Returns:
        List of processed files with metadata
    """
    try:
        files = await memvec_service.list_files()
        return {"files": files, "total": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """
    Get statistics about the MemVec system.
    
    Returns:
        System statistics including vector count, index size, etc.
    """
    try:
        stats = await memvec_service.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
