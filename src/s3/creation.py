"""
S3 Vector creation utilities for MemVec POC.

Simple utilities for:
- Creating S3 buckets
- Creating vector indexes  
- Uploading datasets to S3 vectors
"""

import os
import boto3
import json
import logging
from typing import Dict, Optional
from pathlib import Path
import numpy as np
from ..config.env import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME

logger = logging.getLogger(__name__)


class S3VectorManager:
    """Simple S3 Vector manager for MemVec POC."""
    
    def __init__(self, region_name: Optional[str] = None):
        """Initialize with AWS credentials from environment."""
        self.region_name = region_name or AWS_REGION
        
        try:
            # S3 client for bucket operations
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=self.region_name
            )
            
            # S3 Vectors client for vector operations
            self.s3vectors_client = boto3.client(
                's3vectors',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=self.region_name
            )
            
            logger.info(f"S3 and S3Vectors clients initialized for region: {self.region_name}")
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise
    
    def create_bucket(self, bucket_name: str, use_kms: bool = False, kms_key_arn: str = None) -> bool:
        """Create S3 Vector bucket for vector storage."""
        try:
            payload = {"vectorBucketName": bucket_name}
            
            # Add KMS encryption if requested
            if use_kms:
                if not kms_key_arn:
                    logger.error("KMS Key ARN must be provided when use_kms=True")
                    return False
                payload["encryptionConfiguration"] = {
                    "sseType": "aws:kms",
                    "kmsKeyArn": kms_key_arn
                }
            
            self.s3vectors_client.create_vector_bucket(**payload)
            logger.info(f"Created vector bucket: {bucket_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create vector bucket {bucket_name}: {e}")
            return False
    
    def create_index(self, bucket_name: str, index_name: str, dimension: int, 
                    distance_metric: str = "cosine", non_filterable_keys: list = None) -> bool:
        """Create vector index in S3 Vector bucket."""
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

# Simple convenience functions
def create_bucket_simple(bucket_name: str = None, region: str = None, 
                        use_kms: bool = False, kms_key_arn: str = None) -> bool:
    """Create a vector bucket with default settings."""
    bucket_name = bucket_name or S3_BUCKET_NAME
    manager = S3VectorManager(region_name=region)
    return manager.create_bucket(bucket_name, use_kms, kms_key_arn)


def create_index_simple(bucket_name: str = None, index_name: str = "default", 
                       dimension: int = 384, distance_metric: str = "cosine",
                       non_filterable_keys: list = None) -> bool:
    """Create a vector index with default settings."""
    bucket_name = bucket_name or S3_BUCKET_NAME
    manager = S3VectorManager()
    return manager.create_index(bucket_name, index_name, dimension, distance_metric, non_filterable_keys)


def upload_embedded_simple(bucket_name: str = None, index_name: str = "default", 
                          vectors_data: list = None) -> bool:
    """Upload pre-embedded vectors."""
    if vectors_data is None:
        logger.error("vectors_data is required")
        return False
    bucket_name = bucket_name or S3_BUCKET_NAME
    manager = S3VectorManager()
    return manager.upload_embedded_vectors(bucket_name, index_name, vectors_data)


def upload_texts_simple(bucket_name: str = None, index_name: str = "default", 
                       texts_data: list = None) -> bool:
    """Upload texts to be embedded with Bedrock."""
    if texts_data is None:
        logger.error("texts_data is required")
        return False
    bucket_name = bucket_name or S3_BUCKET_NAME
    manager = S3VectorManager()
    return manager.upload_texts_with_bedrock(bucket_name, index_name, texts_data)


def upload_json_simple(bucket_name: str = None, index_name: str = "default", 
                      json_file: str = None, data_type: str = "embedded") -> bool:
    """Upload vectors from JSON file."""
    if json_file is None:
        logger.error("json_file is required")
        return False
    bucket_name = bucket_name or S3_BUCKET_NAME
    manager = S3VectorManager()
    return manager.upload_dataset_json(bucket_name, index_name, json_file, data_type)