"""
File processing functionality for MemVec.

This module processes files and converts them into Vector objects.
"""

from typing import List, Callable
from sentence_transformers import SentenceTransformer
import PyPDF2
from ..vectors.vectors import Vector
from ..vectors.pointer import generate_document_id


import re
from typing import List
import PyPDF2


def extract_text_from_pdf(file_path: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Extract text chunks from PDF file with sentence-aware splitting."""
    chunks = []
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if not text or not text.strip():
                continue

            # Split into sentences (basic regex: split on ., ?, ! followed by space or end of line)
            sentences = re.split(r'(?<=[.?!])\s+', text.strip())

            current_chunk = []
            current_length = 0
            for sentence in sentences:
                words = sentence.split()
                if current_length + len(words) > chunk_size:
                    # Save current chunk
                    chunks.append(" ".join(current_chunk))

                    # Start a new chunk with overlap
                    if overlap > 0 and current_chunk:
                        overlap_words = " ".join(current_chunk).split()[-overlap:]
                        current_chunk = [" ".join(overlap_words)]
                        current_length = len(overlap_words)
                    else:
                        current_chunk = []
                        current_length = 0

                current_chunk.append(sentence)
                current_length += len(words)

            # Add last chunk if non-empty
            if current_chunk:
                chunks.append(" ".join(current_chunk))

    return chunks



def create_embeddings_with_sentence_transformer(texts: List[str]) -> List[List[float]]:
    """Create embeddings using SentenceTransformer."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return [embedding.tolist() for embedding in embeddings]


def process_file(
    file_path: str,
    text_extractor: Callable[[str], List[str]] = extract_text_from_pdf,
    embedding_generator: Callable[[List[str]], List[List[float]]] = create_embeddings_with_sentence_transformer
) -> List[Vector]:
    """
    Process a file and return Vector objects.
    
    Args:
        file_path: Path to the file to process
        text_extractor: Function to extract text chunks from file
        embedding_generator: Function to generate embeddings from text chunks
        
    Returns:
        List of Vector objects
    """
    # Extract text chunks
    text_chunks = text_extractor(file_path)
    
    # Generate embeddings
    embeddings = embedding_generator(text_chunks)
    
    # Generate document ID for this file
    document_id = generate_document_id()
    
    # Create Vector objects
    vectors = []
    for i, (text, embedding) in enumerate(zip(text_chunks, embeddings)):
        metadata = {
            "source_file": file_path,
            "chunk_index": i,
            "text": text[:200] + "..." if len(text) > 200 else text
        }
        vectors.append(Vector(
            values=embedding, 
            document=document_id,
            chunk=0,  # Single chunk for now
            offset=i,
            metadata=metadata
        ))
    
    return vectors