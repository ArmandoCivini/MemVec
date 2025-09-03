"""
Base classes for file processing components.
"""

from abc import ABC, abstractmethod
from typing import List


class TextExtractor(ABC):
    """Base class for text extraction from files."""
    
    @abstractmethod
    def extract(self, file_path: str) -> List[str]:
        """Extract text chunks from a file."""
        pass


class EmbeddingGenerator(ABC):
    """Base class for generating embeddings from text."""
    
    @abstractmethod
    def generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings from text chunks."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of embeddings this generator produces."""
        pass
