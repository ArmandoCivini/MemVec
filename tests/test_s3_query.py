"""
Test for querying S3 vector bucket to verify embeddings were uploaded correctly.

This test queries the S3 vector bucket with sample queries to check if the
PDF chunks were successfully uploaded and can be retrieved.
"""

import os
import sys
import json
import boto3
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.s3.vector import LocalVector


def test_s3_vector_query(bucket_name: str, index_name: str, region_name: str = "us-east-2"):
    """
    Test function that queries the S3 vector bucket to verify embeddings were uploaded.
    
    Args:
        bucket_name: Name of the S3 vector bucket
        index_name: Name of the vector index
        region_name: AWS region for the clients
    """
    print(f"Querying S3 vector bucket: {bucket_name}")
    print(f"Index: {index_name}")
    print(f"Region: {region_name}")
    print()
    
    # Create S3 Vectors client
    s3vectors = boto3.client("s3vectors", region_name=region_name)
    
    # Test queries related to the PDF content (attention paper)
    test_queries = [
        "attention mechanism in neural networks",
        "transformer architecture", 
        "self-attention computation",
        "query key value matrices",
        "multi-head attention"
    ]
    
    for i, query_text in enumerate(test_queries):
        print(f"Query {i+1}: '{query_text}'")
        print("-" * 50)
        
        # Generate embedding for the query using LocalVector
        query_vector = LocalVector(key="query", text=query_text)
        query_embedding = query_vector.get_embedding()
        
        try:
            # Query vector index
            response = s3vectors.query_vectors(
                vectorBucketName=bucket_name,
                indexName=index_name,
                queryVector={"float32": query_embedding},
                topK=3,
                returnDistance=True,
                returnMetadata=True
            )
            
            print(f"Found {len(response['vectors'])} results:")
            
            for j, result in enumerate(response['vectors']):
                print(f"\nResult {j+1}:")
                print(f"  Key: {result.get('key', 'N/A')}")
                print(f"  Distance: {result.get('distance', 'N/A')}")
                
                metadata = result.get('metadata', {})
                if metadata:
                    print(f"  Metadata:")
                    for key, value in metadata.items():
                        print(f"    {key}: {value}")
                
                # Print a snippet of the actual vector data if available
                data = result.get('data', {})
                if 'float32' in data and len(data['float32']) > 0:
                    vector_preview = data['float32'][:5]  # First 5 dimensions
                    print(f"  Vector preview: {vector_preview}...")
            
        except Exception as e:
            print(f"Error querying with '{query_text}': {str(e)}")
        
        print("\n" + "="*60 + "\n")
    
    # Test a query with metadata filter (if any documents have document_type metadata)
    print("Testing metadata filter query...")
    print("-" * 50)
    
    try:
        query_vector = LocalVector(key="query", text="neural network attention")
        query_embedding = query_vector.get_embedding()
        
        response = s3vectors.query_vectors(
            vectorBucketName=bucket_name,
            indexName=index_name,
            queryVector={"float32": query_embedding},
            topK=5,
            filter={"document_type": "pdf"},
            returnDistance=True,
            returnMetadata=True
        )
        
        print(f"Found {len(response['vectors'])} results with metadata filter 'document_type=pdf':")
        
        for j, result in enumerate(response['vectors']):
            print(f"\nFiltered Result {j+1}:")
            print(f"  Key: {result.get('key', 'N/A')}")
            print(f"  Distance: {result.get('distance', 'N/A')}")
            
            metadata = result.get('metadata', {})
            if metadata:
                print(f"  Metadata:")
                for key, value in metadata.items():
                    print(f"    {key}: {value}")
    
    except Exception as e:
        print(f"Error with metadata filter query: {str(e)}")
    
    print("\n" + "="*60)
    print("Query test completed!")


def test_bucket_info(bucket_name: str, index_name: str, region_name: str = "us-east-2"):
    """
    Get basic information about the S3 vector bucket and index.
    
    Args:
        bucket_name: Name of the S3 vector bucket
        index_name: Name of the vector index
        region_name: AWS region for the clients
    """
    print("Getting bucket and index information...")
    print("-" * 50)
    
    s3vectors = boto3.client("s3vectors", region_name=region_name)
    
    try:
        # List vector buckets
        buckets_response = s3vectors.list_vector_buckets()
        print("Available vector buckets:")
        for bucket in buckets_response.get('vectorBuckets', []):
            print(f"  - {bucket.get('name', 'N/A')}")
        print()
        
        # Describe the specific bucket
        bucket_response = s3vectors.describe_vector_bucket(vectorBucketName=bucket_name)
        print(f"Bucket '{bucket_name}' details:")
        print(f"  Status: {bucket_response.get('status', 'N/A')}")
        print(f"  Creation time: {bucket_response.get('creationTime', 'N/A')}")
        print()
        
        # List indexes in the bucket
        indexes_response = s3vectors.list_vector_indexes(vectorBucketName=bucket_name)
        print(f"Indexes in bucket '{bucket_name}':")
        for index in indexes_response.get('vectorIndexes', []):
            print(f"  - {index.get('name', 'N/A')} (status: {index.get('status', 'N/A')})")
        print()
        
        # Describe the specific index
        index_response = s3vectors.describe_vector_index(
            vectorBucketName=bucket_name,
            indexName=index_name
        )
        print(f"Index '{index_name}' details:")
        print(f"  Status: {index_response.get('status', 'N/A')}")
        print(f"  Dimension: {index_response.get('dimension', 'N/A')}")
        print(f"  Vector count: {index_response.get('vectorCount', 'N/A')}")
        print(f"  Creation time: {index_response.get('creationTime', 'N/A')}")
        
    except Exception as e:
        print(f"Error getting bucket/index info: {str(e)}")


if __name__ == "__main__":
    # Test configuration variables - should match the upload test
    bucket_name = "memvec-test"
    index_name = "memvec-test-index"
    region_name = "us-east-2"
    
    print("S3 Vector Query Test")
    print("===================")
    print()
    
    # First get bucket/index information
    test_bucket_info(bucket_name, index_name, region_name)
    
    print("\n" + "="*60 + "\n")
    
    # Then run the query tests
    test_s3_vector_query(bucket_name, index_name, region_name)