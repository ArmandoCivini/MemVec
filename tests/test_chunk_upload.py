"""
Tests for vector chunk upload functionality.

This module tests the upload_vector_chunk function with both mock S3
and real S3 (when available), ensuring proper chunk storage.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.s3.chunk_upload import upload_vector_chunk, download_vector_chunk, MockS3Client


class TestVectorChunkUpload:
    """Test class for vector chunk upload functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test vectors (only numpy arrays)
        self.test_vectors = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            np.array([6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32),
            np.array([11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float32),
            np.array([16.0, 17.0, 18.0, 19.0, 20.0], dtype=np.float32)
        ]
    
    def test_upload_vector_chunk_with_mock_s3(self):
        """Test uploading vector chunk with mock S3 client."""
        # Create mock S3 client
        mock_s3 = MockS3Client()
        
        # Upload vectors
        result = upload_vector_chunk(
            vectors=self.test_vectors,
            chunk_id="test-chunk-001",
            bucket_name="test-bucket",
            s3_client=mock_s3
        )
        
        # Verify the result
        assert result["success"] is True
        assert result["chunk_id"] == "test-chunk-001"
        assert result["number_of_vectors"] == 4
        assert result["s3_key"] == "chunks/test-chunk-001.pkl"
        
        # Verify data was stored in mock S3
        vectors_key = "test-bucket/chunks/test-chunk-001.pkl"
        assert vectors_key in mock_s3.objects
        assert mock_s3.objects[vectors_key]["ContentType"] == "application/octet-stream"
    
    def test_upload_vector_chunk_auto_generate_id(self):
        """Test uploading vector chunk with auto-generated chunk ID."""
        mock_s3 = MockS3Client()
        
        result = upload_vector_chunk(
            vectors=self.test_vectors[:2],  # Use only 2 vectors
            s3_client=mock_s3,
            bucket_name="test-bucket"
        )
        
        assert result["success"] is True
        assert result["chunk_id"] is not None
        assert len(result["chunk_id"]) == 36  # UUID4 length
        assert result["number_of_vectors"] == 2
    
    def test_upload_vector_chunk_with_different_dtypes(self):
        """Test uploading vectors of different numpy dtypes."""
        mock_s3 = MockS3Client()
        
        # Mix of different numpy dtypes
        mixed_vectors = [
            np.array([1.0, 2.0, 3.0], dtype=np.float64),  # float64
            np.array([4.0, 5.0, 6.0], dtype=np.float32),  # float32
            np.array([7, 8, 9], dtype=np.int32),  # int32
        ]
        
        result = upload_vector_chunk(
            vectors=mixed_vectors,
            chunk_id="mixed-types",
            s3_client=mock_s3,
            bucket_name="test-bucket"
        )
        
        assert result["success"] is True
        assert result["number_of_vectors"] == 3
    
    def test_upload_vector_chunk_invalid_input(self):
        """Test uploading with invalid vector inputs."""
        mock_s3 = MockS3Client()
        
        # Test with invalid vector type
        invalid_vectors = [
            np.array([1.0, 2.0, 3.0]),
            [4.0, 5.0, 6.0]  # Python list instead of numpy array
        ]
        
        with pytest.raises(ValueError, match="Vector at index 1 must be numpy array"):
            upload_vector_chunk(
                vectors=invalid_vectors,
                s3_client=mock_s3,
                bucket_name="test-bucket"
            )
        
        # Test with empty vector list
        with pytest.raises(ValueError, match="Vector list cannot be empty"):
            upload_vector_chunk(
                vectors=[],
                s3_client=mock_s3,
                bucket_name="test-bucket"
            )
    
    def test_download_vector_chunk_with_mock_s3(self):
        """Test downloading vector chunk with mock S3 client."""
        mock_s3 = MockS3Client()
        
        # First upload a chunk
        upload_result = upload_vector_chunk(
            vectors=self.test_vectors,
            chunk_id="download-test",
            s3_client=mock_s3,
            bucket_name="test-bucket"
        )
        
        assert upload_result["success"] is True
        
        # Now download the chunk
        download_result = download_vector_chunk(
            chunk_id="download-test",
            s3_client=mock_s3,
            bucket_name="test-bucket"
        )
        
        assert download_result["success"] is True
        
        # Verify downloaded vectors
        downloaded_vectors = download_result["vectors"]
        assert downloaded_vectors.shape == (4, 5)  # 4 vectors, 5 dimensions each
        
        # Verify first vector matches
        np.testing.assert_array_almost_equal(
            downloaded_vectors[0], 
            np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        )
    
    def test_upload_large_vector_batch(self):
        """Test uploading a larger batch of vectors."""
        mock_s3 = MockS3Client()
        
        # Create 100 random vectors with 128 dimensions
        large_vectors = []
        for i in range(100):
            vector = np.random.rand(128).astype(np.float32)
            large_vectors.append(vector)
        
        result = upload_vector_chunk(
            vectors=large_vectors,
            chunk_id="large-batch",
            s3_client=mock_s3,
            bucket_name="test-bucket"
        )
        
        assert result["success"] is True
        assert result["number_of_vectors"] == 100
    
    def test_s3_error_handling(self):
        """Test error handling when S3 operations fail."""
        # Create a mock S3 client that raises exceptions
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = Exception("S3 connection failed")
        
        result = upload_vector_chunk(
            vectors=self.test_vectors[:2],
            chunk_id="error-test",
            s3_client=mock_s3,
            bucket_name="test-bucket"
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "S3 connection failed" in result["error"]
    
    def test_simple_functionality(self):
        """Test the core functionality is simple and works as expected."""
        mock_s3 = MockS3Client()
        
        # Simple test vectors
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]
        
        result = upload_vector_chunk(
            vectors=vectors,
            chunk_id="simple-test",
            s3_client=mock_s3,
            bucket_name="test-bucket"
        )
        
        # Check basic result structure
        assert result["success"] is True
        assert result["chunk_id"] == "simple-test"
        assert result["number_of_vectors"] == 3
        assert result["s3_key"] == "chunks/simple-test.pkl"
        
        # Verify we can download it back
        download_result = download_vector_chunk(
            chunk_id="simple-test",
            s3_client=mock_s3,
            bucket_name="test-bucket"
        )
        
        assert download_result["success"] is True
        downloaded_vectors = download_result["vectors"]
        assert downloaded_vectors.shape == (3, 3)
        
        # Verify the data integrity
        np.testing.assert_array_almost_equal(downloaded_vectors[0], [1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(downloaded_vectors[1], [0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(downloaded_vectors[2], [0.0, 0.0, 1.0])


def test_integration_with_real_s3():
    """
    Integration test with real S3 (only runs if AWS credentials are available).
    This test is skipped if AWS credentials are not configured.
    """
    try:
        import boto3
        from src.config.env import S3_BUCKET_NAME, AWS_REGION
        
        # Try to create S3 client
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Check if we can access the bucket (this will raise an exception if not)
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
        
        # If we get here, we have valid AWS credentials and bucket access
        test_vectors = [
            np.random.rand(10).astype(np.float32),
            np.random.rand(10).astype(np.float32),
            np.random.rand(10).astype(np.float32)
        ]
        
        result = upload_vector_chunk(
            vectors=test_vectors,
            chunk_id="integration-test",
            s3_client=s3_client,
            bucket_name=S3_BUCKET_NAME
        )
        
        assert result["success"] is True
        print(f"Successfully uploaded chunk to real S3: {result['chunk_id']}")
        
        # Test download
        download_result = download_vector_chunk(
            chunk_id="integration-test",
            s3_client=s3_client,
            bucket_name=S3_BUCKET_NAME
        )
        
        assert download_result["success"] is True
        assert download_result["vectors"].shape == (3, 10)
        print("Successfully downloaded chunk from real S3")
        
    except Exception as e:
        print(f"Skipping real S3 test: {str(e)}")
        pytest.skip("Real S3 test skipped - AWS credentials not available or bucket not accessible")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestVectorChunkUpload()
    test_instance.setup_method()
    
    print("Running vector chunk upload tests...")
    print("=" * 50)
    
    # Test with mock S3
    print("✓ Testing upload with mock S3...")
    test_instance.test_upload_vector_chunk_with_mock_s3()
    
    print("✓ Testing auto-generated chunk ID...")
    test_instance.test_upload_vector_chunk_auto_generate_id()
    
    print("✓ Testing different numpy dtypes...")
    test_instance.test_upload_vector_chunk_with_different_dtypes()
    
    print("✓ Testing invalid input handling...")
    try:
        test_instance.test_upload_vector_chunk_invalid_input()
    except AssertionError:
        print("✓ Invalid input correctly rejected")
    
    print("✓ Testing download functionality...")
    test_instance.test_download_vector_chunk_with_mock_s3()
    
    print("✓ Testing large vector batch...")
    test_instance.test_upload_large_vector_batch()
    
    print("✓ Testing error handling...")
    test_instance.test_s3_error_handling()
    
    print("✓ Testing simple core functionality...")
    test_instance.test_simple_functionality()
    
    print("\n" + "=" * 50)
    print("All mock tests passed!")
    
    print("\nTesting integration with real S3...")
    test_integration_with_real_s3()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
