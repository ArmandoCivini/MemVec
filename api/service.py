"""
Service layer for MemVec API operations.
"""

import asyncio
import tempfile
import os
from typing import List, Dict, Any, Optional
from fastapi import UploadFile
import sys

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.index.index import HNSWIndex
from src.processes.components import SentenceTransformerEmbedding
from src.processes.process_file import FileProcessor
from src.query import query_system


class MemVecService:
    """
    Service layer that orchestrates MemVec operations for the API.
    """
    
    def __init__(self):
        """Initialize MemVec service components."""
        self.embedding_generator = SentenceTransformerEmbedding()
        self.index = HNSWIndex(dimension=self.embedding_generator.dimension)
        self.file_processor = FileProcessor(self.embedding_generator)
        self.bucket_name = "memvec-vectors"  # Default bucket
        
    async def process_file(self, file: UploadFile, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an uploaded file and store vectors in the MemVec system.
        
        Args:
            file: Uploaded file to process
            metadata: Optional metadata to associate with the file
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file.flush()  # Ensure content is written
                
                # Reopen the file in binary mode for processing
                with open(temp_file.name, 'rb') as file_obj:
                    # Process the file using the file processor
                    chunks = self.file_processor.process_file(
                        file_obj=file_obj,
                        filename=file.filename,
                        index=self.index
                    )
                    
                    # Extract vector IDs
                    vector_ids = []
                    for chunk in chunks:
                        for vector in chunk:
                            vector_ids.append(vector.get_id())
                    
                    result = {
                        "success": True,
                        "vector_ids": vector_ids,
                        "chunks_processed": len(chunks)
                    }
                
                if result["success"]:
                    return {
                        "success": True,
                        "filename": file.filename,
                        "file_size": len(content),
                        "vector_ids": result["vector_ids"],
                        "message": f"Successfully processed {file.filename}",
                        "total_vectors": len(result["vector_ids"]),
                        "chunks_processed": result.get("chunks_processed", 0)
                    }
                else:
                    return {
                        "success": False,
                        "filename": file.filename,
                        "file_size": len(content),
                        "vector_ids": [],
                        "message": f"Failed to process {file.filename}: {result.get('error', 'Unknown error')}",
                        "total_vectors": 0,
                        "chunks_processed": 0
                    }
                
                # Clean up temporary file
                os.unlink(temp_file.name)
                
        except Exception as e:
            return {
                "success": False,
                "filename": file.filename or "unknown",
                "file_size": 0,
                "vector_ids": [],
                "message": f"Error processing file: {str(e)}",
                "total_vectors": 0,
                "chunks_processed": 0
            }
    
    async def query_vectors(self, query_text: str, k: int = 5, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Query vectors from the MemVec system.
        
        Args:
            query_text: Text to search for
            k: Number of results to return
            threshold: Optional similarity threshold (filters at index level)
            
        Returns:
            Dictionary with query results
        """
        try:
            # Query the system with threshold passed to index level
            result = query_system(
                query_text=query_text,
                index=self.index,
                bucket_name=self.bucket_name,
                embedding_generator=self.embedding_generator,
                k=k,
                threshold=threshold
            )
            
            if result["success"]:
                search_results = result["search_results"]
                
                return {
                    "query_text": query_text,
                    "search_results": search_results,
                    "total_found": len(search_results),
                    "success": True,
                    "query_embedding": result.get("query_embedding", [])
                }
            else:
                return {
                    "query_text": query_text,
                    "search_results": [],
                    "total_found": 0,
                    "success": False,
                    "query_embedding": None
                }
                
        except Exception as e:
            return {
                "query_text": query_text,
                "search_results": [],
                "total_found": 0,
                "success": False,
                "query_embedding": None,
                "error": str(e)
            }
    
    async def delete_file(self, filename: str) -> bool:
        """
        Delete a file and all its vectors.
        
        Args:
            filename: The filename to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # TODO: Implement file deletion logic
            # 1. Find all vectors belonging to this file
            # 2. Remove vectors from FAISS index
            # 3. Remove vector chunks from S3
            
            return True
            
        except Exception as e:
            return False
    
    async def list_files(self) -> List[Dict[str, Any]]:
        """
        List all processed files.
        
        Returns:
            List of file information dictionaries
        """
        try:
            # This is a simplified implementation
            # In a real system, you'd query S3 or a database for file metadata
            files = []
            
            # For now, return empty list
            # This would be implemented based on how file metadata is stored
            return files
            
        except Exception as e:
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary with system stats
        """
        try:
            return {
                "total_vectors": self.index.get_vector_count() if hasattr(self.index, 'get_vector_count') else 0,
                "total_chunks": 0,  # Would be calculated from S3
                "index_size": self.index.dimension
            }
        except Exception as e:
            return {
                "total_vectors": 0,
                "total_chunks": 0,
                "index_size": 0,
                "error": str(e)
            }
