"""
Simple tests for vector chunk upload functionality.
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.s3.chunk_upload import upload_vector_chunk, download_vector_chunk
from src.s3.mock_client import MockS3Client
from src.s3.creation import create_s3_bucket
from src.vectors.vectors import Vector


def test_chunk_upload_download(use_real_s3=False, bucket_name="test-bucket"):
    """Test uploading and downloading vector chunks."""
    
    # Create test Vector objects
    test_vectors = [
        Vector(id="vec-1", values=[1.0, 2.0, 3.0]),
        Vector(id="vec-2", values=[4.0, 5.0, 6.0]),
        Vector(id="vec-3", values=[7.0, 8.0, 9.0])
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
        chunk_id="test-chunk",
        bucket_name=bucket_name,
        s3_client=s3_client
    )
    
    # Check upload success
    assert upload_result["success"] is True
    assert upload_result["chunk_id"] == "test-chunk"
    assert upload_result["number_of_vectors"] == 3
    print(f"✓ Upload successful: {upload_result['s3_key']}")
    
    # Download chunk
    download_result = download_vector_chunk(
        chunk_id="test-chunk",
        bucket_name=bucket_name,
        s3_client=s3_client
    )
    
    # Check download success
    assert download_result["success"] is True
    downloaded_vectors = download_result["vectors"]
    assert downloaded_vectors.shape == (3, 3)
    print("✓ Download successful")
    
    # Verify data integrity
    assert list(downloaded_vectors[0]) == [1.0, 2.0, 3.0]
    assert list(downloaded_vectors[1]) == [4.0, 5.0, 6.0]
    assert list(downloaded_vectors[2]) == [7.0, 8.0, 9.0]
    print("✓ Data integrity verified")
    
    return True


if __name__ == "__main__":
    # Set this flag to True to test with real S3
    USE_REAL_S3 = False
    # Bucket name for testing
    TEST_BUCKET_NAME = "memvec-chunk-test-bucket"
    
    print("Vector Chunk Upload Test")
    print("=" * 30)
    
    success = test_chunk_upload_download(use_real_s3=USE_REAL_S3, bucket_name=TEST_BUCKET_NAME)
    
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Test failed!")
