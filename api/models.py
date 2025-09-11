"""
Pydantic models for MemVec API requests and responses.
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class FileUploadResponse(BaseModel):
    """Response model for file upload operations."""
    success: bool
    filename: str
    file_size: int
    vector_ids: List[int]
    message: str
    total_vectors: int
    chunks_processed: int


class QueryRequest(BaseModel):
    """Request model for querying vectors."""
    query_text: str
    k: int = 5
    threshold: Optional[float] = None


class QueryResult(BaseModel):
    """Individual query result."""
    vector_values: List[float]
    distance: float
    document_id: int
    chunk_id: int
    offset: int
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    """Response model for query operations."""
    query_text: str
    search_results: List[QueryResult]
    total_found: int
    success: bool
    query_embedding: Optional[List[float]] = None


class SystemStats(BaseModel):
    """System statistics."""
    total_vectors: int
    total_chunks: int
    index_size: int
    cache_hits: int
    cache_misses: int
