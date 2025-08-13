"""
embeddings.py

Handles conversion of text data into vector embeddings for use with MemVec.

Responsibilities:
- Load and manage the embedding model
- Convert raw text to numeric vectors
- Normalize and format vectors for FAISS, Redis, and S3
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", normalize: bool = True):
        """
        Initialize the embedding model.
        Default: SentenceTransformers MiniLM model for speed and small footprint.
        """
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Convert text(s) into vector embeddings.
        
        Args:
            texts: A single string or a list of strings.
        
        Returns:
            np.ndarray: Embeddings as a 2D NumPy array (shape: n_samples x dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )

        # FAISS works best with float32
        return np.array(embeddings, dtype=np.float32)

    def get_dimension(self) -> int:
        """
        Get the dimensionality of the embeddings produced by this model.
        Useful for initializing FAISS indexes.
        """
        dummy = self.encode("test")
        return dummy.shape[1]
