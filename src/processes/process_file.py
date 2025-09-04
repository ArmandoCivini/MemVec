"""
File processing functionality for MemVec.

This module processes files and converts them into Vector objects.
"""

from typing import List
from ..vectors.vectors import Vector
from ..vectors.pointer import generate_document_id
from ..index.index import HNSWIndex
from ..config.contants import MAX_VECTORS_PER_CHUNK, METADATA_TEXT_PREVIEW_LENGTH
from .base import TextExtractor, EmbeddingGenerator
from .components import PDFTextExtractor, SentenceTransformerEmbedding


class FileProcessor:
    """Process files and add vectors to an index."""
    
    def __init__(self, text_extractor: TextExtractor, embedding_generator: EmbeddingGenerator):
        self.text_extractor = text_extractor
        self.embedding_generator = embedding_generator
    
    def get_index_dimension(self) -> int:
        """Get the required index dimension based on the embedding generator."""
        return self.embedding_generator.dimension
    
    def process_file(self, file_path: str, index: HNSWIndex) -> List[List[Vector]]:
        """
        Process a file and return Vector objects with proper chunking.
        Adds vectors to the provided index.
        
        Args:
            file_path: Path to the file to process
            index: HNSW index to add vectors to
            
        Returns:
            List of lists of Vector objects, where each inner list represents one chunk
        """
        # Extract text chunks
        text_chunks = self.text_extractor.extract(file_path)
        
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
                "source_file": file_path,
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
def process_file(file_path: str, index: HNSWIndex) -> List[List[Vector]]:
    """
    Process a file using default components.
    
    Args:
        file_path: Path to the file to process
        index: HNSW index to add vectors to
        
    Returns:
        List of lists of Vector objects, where each inner list represents one chunk
    """
    text_extractor = PDFTextExtractor()
    embedding_generator = SentenceTransformerEmbedding()
    processor = FileProcessor(text_extractor, embedding_generator)
    return processor.process_file(file_path, index)