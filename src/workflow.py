"""
Complete workflow for adding files to the MemVec system.

This module provides the end-to-end functionality for processing files,
adding them to the index, and storing chunks in S3.
"""

from typing import List, Dict, Any
from .processes.process_file import FileProcessor
from .index.index import HNSWIndex
from .s3.chunk_upload import upload_vector_chunk
from .vectors.vectors import Vector


def add_file_to_system(
    file_path: str,
    processor: FileProcessor,
    index: HNSWIndex,
    bucket_name: str,
    s3_client=None
) -> Dict[str, Any]:
    """
    Complete workflow to add a file to the MemVec system.
    
    1. Process the file into vector chunks
    2. Add vectors to the index
    3. Upload each chunk to S3
    
    Args:
        file_path: Path to the file to process
        processor: FileProcessor instance with configured components
        index: HNSW index to add vectors to
        bucket_name: S3 bucket name for storing chunks
        s3_client: Optional S3 client
        
    Returns:
        Dictionary containing:
        - document_id: The document ID assigned to this file
        - total_vectors: Total number of vectors created
        - chunks_count: Number of chunks created
        - s3_uploads: List of upload results for each chunk
        - success: Boolean indicating overall success
    """
    try:
        # Process the file into chunks using file object
        filename = file_path.split('/')[-1]  # Extract filename from path
        with open(file_path, 'rb') as file_obj:
            chunks = processor.process_file(file_obj, filename, index)
        
        # Get document ID from first vector
        document_id = chunks[0][0].document if chunks and chunks[0] else None
        total_vectors = sum(len(chunk) for chunk in chunks)
        
        # Upload each chunk to S3
        s3_uploads = []
        for chunk_idx, chunk in enumerate(chunks):
            # Get chunk ID from the first vector in the chunk
            chunk_id = chunk[0].get_chunk_id() if chunk else None
            
            upload_result = upload_vector_chunk(
                vectors=chunk,
                chunk_id=chunk_id,
                bucket_name=bucket_name,
                s3_client=s3_client
            )
            s3_uploads.append(upload_result)
        
        # Check if all uploads succeeded
        all_uploads_successful = all(upload["success"] for upload in s3_uploads)
        
        return {
            "document_id": document_id,
            "total_vectors": total_vectors,
            "chunks_count": len(chunks),
            "s3_uploads": s3_uploads,
            "success": all_uploads_successful
        }
        
    except Exception as e:
        return {
            "document_id": None,
            "total_vectors": 0,
            "chunks_count": 0,
            "s3_uploads": [],
            "success": False,
            "error": str(e)
        }
