"""
S3 vector append operations for MemVec.

This module provides functionality to append/insert vector data
into S3 vector buckets using the S3 Vectors API.
"""

import boto3
from typing import List
from .vector import Vector


def append_vectors_to_bucket(bucket: str, index: str, data_list: List[Vector], region_name: str = "us-west-2"):
    """
    Insert a list of Vector instances into an S3 vector bucket.
    
    Based on the AWS S3 Vectors example code pattern.
    
    Args:
        bucket: Name of the S3 vector bucket
        index: Name of the vector index
        data_list: List of Vector instances to insert
        region_name: AWS region for the S3 Vectors client
    """
    # Create S3 Vectors client
    s3vectors = boto3.client("s3vectors", region_name=region_name)
    
    # Convert Vector instances to S3 vectors format
    vectors = []
    for vector in data_list:
        s3_vector = vector.to_s3_vector_format()
        vectors.append(s3_vector)
    
    # Write embeddings into vector index with metadata
    s3vectors.put_vectors(
        vectorBucketName=bucket,   
        indexName=index,   
        vectors=vectors
    )