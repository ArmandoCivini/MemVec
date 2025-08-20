"""
Vector class for S3 vector operations.

This module provides a Vector class that encapsulates vector data
and handles embedding generation for S3 vector storage.
"""

import boto3
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, Optional, List


class Vector:
    """
    Base class for vectors with key, data, and metadata for S3 vector storage.
    """
    
    def __init__(self, key: str, text: Optional[str] = None, 
                 embedding: Optional[List[float]] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a Vector instance.
        
        Args:
            key: Unique identifier for the vector
            text: Text to be embedded (mutually exclusive with embedding)
            embedding: Pre-computed embedding (mutually exclusive with text)
            metadata: Optional metadata dictionary
        """
        self.key = key
        self.text = text
        self._embedding = embedding
        self.metadata = metadata or {}
        
        if text is None and embedding is None:
            raise ValueError("Either text or embedding must be provided")
        if text is not None and embedding is not None:
            raise ValueError("Cannot provide both text and embedding")
    
    def generate_embedding(self) -> List[float]:
        """
        Generate embedding from text. Must be implemented by child classes.
        
        Returns:
            List of floats representing the embedding
        """
        raise NotImplementedError("Child classes must implement generate_embedding()")
    
    def get_embedding(self) -> List[float]:
        """
        Get the embedding, generating it if necessary.
        
        Returns:
            List of floats representing the embedding
        """
        if self._embedding is not None:
            return self._embedding
        
        if self.text is not None:
            self._embedding = self.generate_embedding()
        
        return self._embedding
    
    def to_s3_vector_format(self) -> Dict[str, Any]:
        """
        Convert to the format expected by S3 Vectors put_vectors API.
        
        Returns:
            Dictionary in S3 vectors format
        """
        embedding = self.get_embedding()
        
        vector_obj = {
            "key": self.key,
            "data": {"float32": embedding}
        }
        
        if self.metadata:
            vector_obj["metadata"] = self.metadata
        
        return vector_obj


class LocalVector(Vector):
    """
    Vector class that uses local sentence-transformers for embedding generation.
    """
    
    def generate_embedding(self) -> List[float]:
        """
        Generate 1024-dimensional embedding from text using local model.
        
        Returns:
            List of floats representing the embedding
        """
        if self.text is None:
            raise ValueError("Cannot generate embedding without text")
        
        # Use a model that produces 1024-dimensional embeddings
        model = SentenceTransformer('all-mpnet-base-v2')  # 768 dims
        embedding = model.encode(self.text, convert_to_numpy=True)
        
        # Pad or truncate to 1024 dimensions
        if len(embedding) < 1024:
            # Pad with zeros
            padded = np.zeros(1024)
            padded[:len(embedding)] = embedding
            embedding = padded
        elif len(embedding) > 1024:
            # Truncate
            embedding = embedding[:1024]
        
        return embedding.tolist()


class BedrockVector(Vector):
    """
    Vector class that uses Amazon Bedrock for embedding generation.
    """
    
    def __init__(self, key: str, text: Optional[str] = None, 
                 embedding: Optional[List[float]] = None, 
                 metadata: Optional[Dict[str, Any]] = None,
                 region_name: str = "us-west-2",
                 model_id: str = "amazon.titan-embed-text-v2:0"):
        """
        Initialize a BedrockVector instance.
        
        Args:
            key: Unique identifier for the vector
            text: Text to be embedded (mutually exclusive with embedding)
            embedding: Pre-computed embedding (mutually exclusive with text)
            metadata: Optional metadata dictionary
            region_name: AWS region for Bedrock client
            model_id: Bedrock model ID for embedding generation
        """
        super().__init__(key, text, embedding, metadata)
        self.region_name = region_name
        self.model_id = model_id
        self._bedrock_client = None
    
    @property
    def bedrock_client(self):
        """Lazy initialization of Bedrock client."""
        if self._bedrock_client is None:
            self._bedrock_client = boto3.client("bedrock-runtime", region_name=self.region_name)
        return self._bedrock_client
    
    def generate_embedding(self) -> List[float]:
        """
        Generate embedding from text using Amazon Bedrock.
        
        Returns:
            List of floats representing the embedding
        """
        if self.text is None:
            raise ValueError("Cannot generate embedding without text")
        
        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({"inputText": self.text})
        )
        
        response_body = json.loads(response["body"].read())
        return response_body["embedding"]
