"""
Simple tests for vector chunk delete functionality.
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.s3.chunk_upload import upload_vector_chunk, download_vector_chunk
from src.s3.delete import delete_vector_chunk
from src.s3.mock_client import MockS3Client
from src.s3.creation import create_s3_bucket
from src.vectors.vectors import Vector


def test_chunk_delete(use_real_s3=False, bucket_name="test-bucket"):
    """Test uploading, downloading, and deleting vector chunks."""
    
    # Create test Vector objects
    test_vectors = [
        Vector(values=[1.0, 2.0, 3.0], document=1, chunk=0, offset=0),
        Vector(values=[4.0, 5.0, 6.0], document=1, chunk=0, offset=1),
        Vector(values=[7.0, 8.0, 9.0], document=1, chunk=0, offset=2)
    ]
    
    # Choose S3 client
    if use_real_s3:
        try:
            import boto3
            from src.config.env import AWS_REGION
            s3_client = boto3.client('s3', region_name=AWS_REGION)
            print("Using real S3...")
            
            # Create bucket if it doesn't exist
            bucket_result = create_s3_bucket(
                bucket_name=bucket_name,
                region=AWS_REGION,
                s3_client=s3_client
            )
            
            if bucket_result["success"]:
                if bucket_result["created"]:
                    print(f"✓ Created bucket: {bucket_name}")
                else:
                    print(f"✓ Bucket already exists: {bucket_name}")
            else:
                print(f"✗ Failed to create/check bucket: {bucket_result['error']}")
                return False
                
        except Exception as e:
            print(f"Cannot use real S3: {e}")
            return False
    else:
        s3_client = MockS3Client()
        print("Using mock S3...")
    
    # Upload chunk
    upload_result = upload_vector_chunk(
        vectors=test_vectors,
        chunk_id="test-delete-chunk",
        bucket_name=bucket_name,
        s3_client=s3_client
    )
    
    assert upload_result["success"] is True
    print(f"✓ Upload successful: {upload_result['s3_key']}")
    
    # Verify chunk exists by downloading
    download_result = download_vector_chunk(
        chunk_id="test-delete-chunk",
        bucket_name=bucket_name,
        s3_client=s3_client
    )
    
    assert download_result["success"] is True
    print("✓ Chunk exists and can be downloaded")
    
    # Delete chunk
    delete_result = delete_vector_chunk(
        chunk_id="test-delete-chunk",
        bucket_name=bucket_name,
        s3_client=s3_client
    )
    
    assert delete_result["success"] is True
    assert delete_result["chunk_id"] == "test-delete-chunk"
    print(f"✓ Delete successful: {delete_result['s3_key']}")
    
    # Verify chunk no longer exists
    download_result_after_delete = download_vector_chunk(
        chunk_id="test-delete-chunk",
        bucket_name=bucket_name,
        s3_client=s3_client
    )
    
    assert download_result_after_delete["success"] is False
    print("✓ Chunk no longer exists after deletion")


def test_chunk_delete_real_s3():
    """Test with real S3 if configured."""
    test_chunk_delete(use_real_s3=True, bucket_name="memvec-delete-test-bucket")


def test_chunk_delete_mock():
    """Test with mocked S3."""
    test_chunk_delete(use_real_s3=False, bucket_name="memvec-delete-test-bucket")



