"""
Concrete implementations of text extraction and embedding generation.
"""

import re
from typing import List, BinaryIO
import PyPDF2
from sentence_transformers import SentenceTransformer

from .base import TextExtractor, EmbeddingGenerator
from ..config.contants import DEFAULT_TEXT_CHUNK_SIZE, DEFAULT_TEXT_OVERLAP


class PDFTextExtractor(TextExtractor):
    """Extract text from PDF files with sentence-aware chunking."""
    
    def __init__(self, chunk_size: int = DEFAULT_TEXT_CHUNK_SIZE, overlap: int = DEFAULT_TEXT_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def extract(self, file_obj: BinaryIO, filename: str = None) -> List[str]:
        """Extract text chunks from PDF file object with sentence-aware splitting."""
        chunks = []
        reader = PyPDF2.PdfReader(file_obj)
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
                if current_length + len(words) > self.chunk_size:
                    # Save current chunk
                    chunks.append(" ".join(current_chunk))

                    # Start a new chunk with overlap
                    if self.overlap > 0 and current_chunk:
                        overlap_words = " ".join(current_chunk).split()[-self.overlap:]
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


class SentenceTransformerEmbedding(EmbeddingGenerator):
    """Generate embeddings using SentenceTransformer models."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    def generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using SentenceTransformer."""
        embeddings = self.model.encode(texts)
        return [embedding.tolist() for embedding in embeddings]
    
    @property
    def dimension(self) -> int:
        """Return the dimension of embeddings this generator produces."""
        return self._dimension
