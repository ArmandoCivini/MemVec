"""
Simple test for S3 vector creation using PDF documents.

This test reads a PDF file, extracts text chunks, creates Vector instances,
and populates an S3 vector bucket using the append functionality.
"""

import os
import sys
import pytest
from pathlib import Path
import PyPDF2

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.s3.vector import LocalVector
from src.s3.append import append_vectors_to_bucket


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    return text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at word boundary
        if end < len(text):
            last_space = chunk.rfind(' ')
            if last_space != -1:
                chunk = chunk[:last_space]
                end = start + last_space
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return [chunk for chunk in chunks if chunk]  # Remove empty chunks


def test_pdf_to_s3_vectors(pdf_path: str, bucket_name: str, index_name: str, region_name: str = "us-west-2"):
    """
    Test function that takes a PDF path, bucket name, and index name
    and populates the S3 bucket with vector data from the PDF.
    
    Args:
        pdf_path: Path to the PDF file to process
        bucket_name: Name of the S3 vector bucket
        index_name: Name of the vector index
        region_name: AWS region for the clients
    """
    
    # Check if PDF file exists
    if not Path(pdf_path).exists():
        pytest.skip(f"PDF file not found at {pdf_path}")
    
    print(f"Processing PDF: {pdf_path}")
    print(f"Target bucket: {bucket_name}")
    print(f"Target index: {index_name}")
    
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(pdf_text)} characters from PDF")
    
    # Chunk the text into manageable pieces
    text_chunks = chunk_text(pdf_text, chunk_size=800, overlap=50)
    print(f"Created {len(text_chunks)} text chunks")
    
    # Create Vector instances for each chunk
    vectors = []
    for i, chunk in enumerate(text_chunks):
        vector = LocalVector(
            key=f"pdf_chunk_{i:04d}",
            text=chunk,
            metadata={
                "source_file": Path(pdf_path).name,
                "chunk_index": i,
                "chunk_length": len(chunk),
                "document_type": "pdf"
            }
        )
        vectors.append(vector)
    
    print(f"Created {len(vectors)} Vector instances")
    
    # Populate S3 bucket with vectors
    try:
        append_vectors_to_bucket(
            bucket=bucket_name,
            index=index_name,
            data_list=vectors,
            region_name=region_name
        )
        print(f"Successfully populated bucket '{bucket_name}' with {len(vectors)} vectors")
        
    except Exception as e:
        pytest.fail(f"Failed to populate S3 bucket: {str(e)}")


if __name__ == "__main__":
    # Test configuration variables
    pdf_path = "datasets/attention.pdf"
    bucket_name = "memvec-test"
    index_name = "memvec-test-index"
    region_name = "us-east-1"
    
    # Run the test with the specified parameters
    test_pdf_to_s3_vectors(pdf_path, bucket_name, index_name, region_name)