"""
S3 Vector creation utilities for MemVec POC.

Note: S3 Vector buckets must be created via AWS Console/CLI first.
This module handles index creation and data upload to existing buckets.
"""

import os
import boto3
import json
import logging
from typing import Dict, Optional, List
from pathlib import Path
import numpy as np
from ..config.env import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME

logger = logging.getLogger(__name__)


class S3VectorManager:
    """S3 Vector manager for existing vector buckets."""
    
    def __init__(self, region_name: Optional[str] = None):
        """Initialize with AWS credentials from environment."""
        self.region_name = region_name or AWS_REGION
        
        try:
            # Only need s3vectors client since buckets are pre-created
            self.s3vectors_client = boto3.client(
                's3vectors',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=self.region_name
            )
            
            logger.info(f"S3Vectors client initialized for region: {self.region_name}")
        except Exception as e:
            logger.error(f"Failed to initialize s3vectors client: {e}")
            raise
    
    def list_buckets(self) -> List[str]:
        """List available vector buckets."""
        try:
            response = self.s3vectors_client.list_vector_buckets()
            buckets = [bucket['vectorBucketName'] for bucket in response.get('vectorBuckets', [])]
            logger.info(f"Found {len(buckets)} vector buckets: {buckets}")
            return buckets
        except Exception as e:
            logger.error(f"Failed to list vector buckets: {e}")
            return []
    
    def bucket_exists(self, bucket_name: str) -> bool:
        """Check if vector bucket exists."""
        try:
            buckets = self.list_buckets()
            return bucket_name in buckets
        except Exception as e:
            logger.error(f"Error checking bucket existence: {e}")
            return False
    
    def list_indexes(self, bucket_name: str) -> List[str]:
        """List indexes in a vector bucket."""
        try:
            response = self.s3vectors_client.list_indexes(vectorBucketName=bucket_name)
            indexes = [idx['indexName'] for idx in response.get('indexes', [])]
            logger.info(f"Found {len(indexes)} indexes in bucket '{bucket_name}': {indexes}")
            return indexes
        except Exception as e:
            logger.error(f"Failed to list indexes in bucket {bucket_name}: {e}")
            return []
    
    def index_exists(self, bucket_name: str, index_name: str) -> bool:
        """Check if index exists in bucket."""
        try:
            indexes = self.list_indexes(bucket_name)
            return index_name in indexes
        except Exception as e:
            logger.error(f"Error checking index existence: {e}")
            return False
    def create_index(self, bucket_name: str, index_name: str, dimension: int, 
                    distance_metric: str = "cosine", non_filterable_keys: list = None) -> bool:
        """Create vector index in existing S3 Vector bucket."""
        if not self.bucket_exists(bucket_name):
            logger.error(f"Vector bucket '{bucket_name}' does not exist. Create it via AWS Console first.")
            return False
            
        if self.index_exists(bucket_name, index_name):
            logger.info(f"Index '{index_name}' already exists in bucket '{bucket_name}'")
            return True
            
        try:
            payload = {
                "vectorBucketName": bucket_name,
                "indexName": index_name,
                "dataType": "float32",
                "dimension": dimension,
                "distanceMetric": distance_metric,
            }
            
            # Add metadata configuration if provided
            if non_filterable_keys:
                payload["metadataConfiguration"] = {
                    "nonFilterableMetadataKeys": non_filterable_keys
                }
            
            self.s3vectors_client.create_index(**payload)
            logger.info(f"Created vector index '{index_name}' in bucket '{bucket_name}' with dimension {dimension}")
            return True
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            return False
    
    def delete_index(self, bucket_name: str, index_name: str) -> bool:
        """Delete vector index from bucket."""
        try:
            self.s3vectors_client.delete_index(
                vectorBucketName=bucket_name,
                indexName=index_name
            )
            logger.info(f"Deleted index '{index_name}' from bucket '{bucket_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete index {index_name}: {e}")
            return False
    
    def upload_embedded_vectors(self, bucket_name: str, index_name: str, 
                               vectors_data: list) -> bool:
        """
        Upload pre-embedded vectors to S3 Vector index using AWS S3 Vectors API.
        
        Args:
            bucket_name: S3 bucket name
            index_name: Vector index name
            vectors_data: List of vector objects in AWS S3 Vectors format:
                [
                    {
                        "key": "unique_id",
                        "data": {"float32": [0.1, 0.2, ...]},  # embedding as list
                        "metadata": {"field1": "value1", ...}  # optional
                    },
                    ...
                ]
        
        Returns:
            bool: True if upload successful
        """
        try:
            self.s3vectors_client.put_vectors(
                vectorBucketName=bucket_name,
                indexName=index_name,
                vectors=vectors_data
            )
            
            logger.info(f"Uploaded {len(vectors_data)} vectors to index '{index_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to upload embedded vectors: {e}")
            return False
    
    def upload_texts_with_bedrock(self, bucket_name: str, index_name: str,
                                 texts_data: list, model_id: str = "amazon.titan-embed-text-v2:0") -> bool:
        """
        Upload text data by first embedding it with Amazon Bedrock, then storing in S3 Vectors.
        
        Args:
            bucket_name: S3 bucket name
            index_name: Vector index name
            texts_data: List of text objects to embed:
                [
                    {
                        "key": "unique_id",
                        "text": "text to embed",
                        "metadata": {"field1": "value1", ...}  # optional
                    },
                    ...
                ]
            model_id: Bedrock embedding model ID
            
        Returns:
            bool: True if upload successful
        """
        try:
            # Initialize Bedrock client
            bedrock_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=self.region_name
            )
            
            # Generate embeddings and prepare vectors
            vectors_data = []
            for item in texts_data:
                # Generate embedding
                response = bedrock_client.invoke_model(
                    modelId=model_id,
                    body=json.dumps({"inputText": item["text"]})
                )
                
                # Extract embedding
                response_body = json.loads(response["body"].read())
                embedding = response_body["embedding"]
                
                # Prepare vector object for S3 Vectors
                vector_obj = {
                    "key": item["key"],
                    "data": {"float32": embedding}
                }
                
                # Add metadata if provided
                if "metadata" in item:
                    vector_obj["metadata"] = item["metadata"]
                
                vectors_data.append(vector_obj)
            
            # Upload to S3 Vectors
            return self.upload_embedded_vectors(bucket_name, index_name, vectors_data)
            
        except Exception as e:
            logger.error(f"Failed to upload texts with Bedrock embedding: {e}")
            return False
    
    def upload_dataset_json(self, bucket_name: str, index_name: str, 
                           json_file: str, data_type: str = "embedded") -> bool:
        """
        Upload vectors from JSON file.
        
        Args:
            bucket_name: S3 bucket name
            index_name: Vector index name
            json_file: Path to JSON file
            data_type: "embedded" for pre-embedded data, "text" for text to be embedded
            
        JSON format for embedded data:
            [
                {
                    "key": "id1",
                    "data": {"float32": [0.1, 0.2, ...]},
                    "metadata": {...}  # optional
                },
                ...
            ]
            
        JSON format for text data:
            [
                {
                    "key": "id1", 
                    "text": "text to embed",
                    "metadata": {...}  # optional
                },
                ...
            ]
        """
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if data_type == "embedded":
                return self.upload_embedded_vectors(bucket_name, index_name, data)
            elif data_type == "text":
                return self.upload_texts_with_bedrock(bucket_name, index_name, data)
            else:
                logger.error(f"Invalid data_type: {data_type}. Use 'embedded' or 'text'")
                return False
                
        except Exception as e:
            logger.error(f"Failed to upload dataset from {json_file}: {e}")
            return False

# Convenience functions that work with existing buckets
def ensure_bucket_exists(bucket_name: str = None) -> str:
    """Verify bucket exists and return its name, or show available buckets."""
    bucket_name = bucket_name or S3_BUCKET_NAME
    manager = S3VectorManager()
    
    if manager.bucket_exists(bucket_name):
        logger.info(f"Using existing vector bucket: {bucket_name}")
        return bucket_name
    else:
        available_buckets = manager.list_buckets()
        if available_buckets:
            logger.error(f"Bucket '{bucket_name}' not found. Available buckets: {available_buckets}")
            logger.error("Please create the bucket via AWS Console or use an existing one.")
        else:
            logger.error("No vector buckets found. Please create one via AWS Console first.")
        return None


def create_index_simple(bucket_name: str = None, index_name: str = "default", 
                       dimension: int = 384, distance_metric: str = "cosine",
                       non_filterable_keys: list = None) -> bool:
    """Create a vector index in existing bucket."""
    bucket_name = ensure_bucket_exists(bucket_name)
    if not bucket_name:
        return False
        
    manager = S3VectorManager()
    return manager.create_index(bucket_name, index_name, dimension, distance_metric, non_filterable_keys)


def upload_embedded_simple(bucket_name: str = None, index_name: str = "default", 
                          vectors_data: list = None) -> bool:
    """Upload pre-embedded vectors to existing bucket."""
    if vectors_data is None:
        logger.error("vectors_data is required")
        return False
        
    bucket_name = ensure_bucket_exists(bucket_name)
    if not bucket_name:
        return False
        
    manager = S3VectorManager()
    return manager.upload_embedded_vectors(bucket_name, index_name, vectors_data)


def upload_texts_simple(bucket_name: str = None, index_name: str = "default", 
                       texts_data: list = None) -> bool:
    """Upload texts to be embedded with Bedrock."""
    if texts_data is None:
        logger.error("texts_data is required")
        return False
        
    bucket_name = ensure_bucket_exists(bucket_name)
    if not bucket_name:
        return False
        
    manager = S3VectorManager()
    return manager.upload_texts_with_bedrock(bucket_name, index_name, texts_data)


def upload_json_simple(bucket_name: str = None, index_name: str = "default", 
                      json_file: str = None, data_type: str = "embedded") -> bool:
    """Upload vectors from JSON file to existing bucket."""
    if json_file is None:
        logger.error("json_file is required")
        return False
        
    bucket_name = ensure_bucket_exists(bucket_name)
    if not bucket_name:
        return False
        
    manager = S3VectorManager()
    return manager.upload_dataset_json(bucket_name, index_name, json_file, data_type)