"""
File processing functionality for MemVec.

This module processes files and converts them into Vector objects.
"""

from typing import List, BinaryIO
from ..vectors.vectors import Vector
from ..vectors.pointer import generate_document_id
from ..index.index import HNSWIndex
from ..config.contants import MAX_VECTORS_PER_CHUNK, METADATA_TEXT_PREVIEW_LENGTH
from .base import TextExtractor, EmbeddingGenerator
from .components import PDFTextExtractor, SentenceTransformerEmbedding


class FileProcessor:
    """Process files and add vectors to an index."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.text_extractor = PDFTextExtractor()
        self.embedding_generator = embedding_generator
    
    def get_index_dimension(self) -> int:
        """Get the required index dimension based on the embedding generator."""
        return self.embedding_generator.dimension
    
    def process_file(self, file_obj: BinaryIO, filename: str, index: HNSWIndex) -> List[List[Vector]]:
        """
        Process a file object and return Vector objects with proper chunking.
        Adds vectors to the provided index.
        
        Args:
            file_obj: File object to process
            filename: Original filename
            index: HNSW index to add vectors to
            
        Returns:
            List of lists of Vector objects, where each inner list represents one chunk
        """
        # Extract text chunks
        text_chunks = self.text_extractor.extract(file_obj, filename)
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate(text_chunks)
        
        # Generate document ID for this file
        document_id = generate_document_id()
        
        # Create Vector objects with proper chunking
        chunks = []  # List of lists of vectors
        current_chunk_vectors = []  # Current chunk being built
        current_chunk = 0
        current_offset = 0
        
        for i, (text, embedding) in enumerate(zip(text_chunks, embeddings)):
            # Check if we need to move to next chunk
            if current_offset >= MAX_VECTORS_PER_CHUNK:
                # Save current chunk and start new one
                chunks.append(current_chunk_vectors)
                current_chunk_vectors = []
                current_chunk += 1
                current_offset = 0
            
            metadata = {
                "source_file": filename,
                "text_index": i,  # Original index in the text chunks
                "text": text[:METADATA_TEXT_PREVIEW_LENGTH] + "..." if len(text) > METADATA_TEXT_PREVIEW_LENGTH else text
            }
            
            vector = Vector(
                values=embedding, 
                document=document_id,
                chunk=current_chunk,
                offset=current_offset,
                metadata=metadata
            )
            
            current_chunk_vectors.append(vector)
            current_offset += 1
        
        # Add the last chunk if it has vectors
        if current_chunk_vectors:
            chunks.append(current_chunk_vectors)
        
        # Flatten all vectors for adding to index
        all_vectors = [vector for chunk in chunks for vector in chunk]
        index.add_vectors(all_vectors)
        
        return chunks


# Convenience function for backward compatibility
def process_file_to_vectors(file_path: str, index: HNSWIndex, bucket_name: str, embedding_generator) -> dict:
    """
    Process a file using file path (for backward compatibility).
    
    Args:
        file_path: Path to the file to process
        index: HNSW index to add vectors to
        bucket_name: S3 bucket name (unused in this implementation)
        embedding_generator: Embedding generator to use
        
    Returns:
        Dictionary with processing results
    """
    try:
        processor = FileProcessor(embedding_generator)
        
        with open(file_path, 'rb') as file:
            filename = file_path.split('/')[-1]
            chunks = processor.process_file(file, filename, index)
        
        # Extract vector IDs
        vector_ids = []
        for chunk in chunks:
            for vector in chunk:
                vector_ids.append(vector.get_id())
        
        return {
            "success": True,
            "vector_ids": vector_ids,
            "chunks_processed": len(chunks)
        }
    except Exception as e:
        return {
            "success": False,
            "vector_ids": [],
            "chunks_processed": 0,
            "error": str(e)
        }