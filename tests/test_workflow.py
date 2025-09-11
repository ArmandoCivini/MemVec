"""
Tests for the complete file processing workflow.
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.workflow import add_file_to_system
from src.processes.process_file import FileProcessor
from src.processes.components import SentenceTransformerEmbedding
from src.index.index import HNSWIndex
from src.s3.mock_client import MockS3Client
from src.vectors.pointer import Pointer
from src.vectors.vectors import Vector


def test_vector_chunk_id_method():
    """Test the new get_chunk_id method in Vector class."""
    
    print("Testing Vector.get_chunk_id() method...")
    
    # Create a test vector
    document_id = 12345
    chunk_number = 67
    offset = 10
    
    vector = Vector(
        values=[1.0, 2.0, 3.0],
        document=document_id,
        chunk=chunk_number,
        offset=offset
    )
    
    # Get chunk ID from vector
    chunk_id = vector.get_chunk_id()
    
    # Verify it matches the direct Pointer method
    expected_chunk_id = Pointer.generate_chunk_id(document_id, chunk_number)
    assert chunk_id == expected_chunk_id, f"Vector chunk ID {chunk_id} should match Pointer chunk ID {expected_chunk_id}"
    
    # Verify it can be decoded correctly
    decoded_doc, decoded_chunk = Pointer.decode_chunk_id(chunk_id)
    assert decoded_doc == document_id, f"Decoded document should be {document_id}, got {decoded_doc}"
    assert decoded_chunk == chunk_number, f"Decoded chunk should be {chunk_number}, got {decoded_chunk}"
    
    print(f"✓ Vector.get_chunk_id() works correctly")
    print(f"  Vector: doc={document_id}, chunk={chunk_number}, offset={offset}")
    print(f"  Chunk ID: {chunk_id}")
    print(f"  Decoded: doc={decoded_doc}, chunk={decoded_chunk}")


def test_chunk_id_generation():
    """Test the new chunk ID generation methods."""
    
    print("Testing chunk ID generation...")
    
    # Test encoding and decoding
    document_id = 12345
    chunk_number = 67
    
    # Generate chunk ID
    chunk_id = Pointer.generate_chunk_id(document_id, chunk_number)
    
    # Decode it back
    decoded_doc, decoded_chunk = Pointer.decode_chunk_id(chunk_id)
    
    assert decoded_doc == document_id, f"Document ID should be {document_id}, got {decoded_doc}"
    assert decoded_chunk == chunk_number, f"Chunk number should be {chunk_number}, got {decoded_chunk}"
    
    print(f"✓ Chunk ID generation works correctly")
    print(f"  Document: {document_id}, Chunk: {chunk_number}")
    print(f"  Generated chunk ID: {chunk_id}")
    print(f"  Decoded back to: {decoded_doc}, {decoded_chunk}")
    
    # Test edge cases
    # Test with maximum values
    max_limits = Pointer.get_limits()
    max_doc = max_limits["max_document"]
    max_chunk = max_limits["max_chunk"]
    
    max_chunk_id = Pointer.generate_chunk_id(max_doc, max_chunk)
    decoded_max_doc, decoded_max_chunk = Pointer.decode_chunk_id(max_chunk_id)
    
    assert decoded_max_doc == max_doc, f"Max document ID should be {max_doc}, got {decoded_max_doc}"
    assert decoded_max_chunk == max_chunk, f"Max chunk should be {max_chunk}, got {decoded_max_chunk}"
    
    print(f"✓ Edge case testing passed with max values: doc={max_doc}, chunk={max_chunk}")


def test_add_file_to_system_mock_s3(test_file_path="datasets/attention.pdf"):
    """Test the complete workflow with mocked S3."""
    
    print(f"Testing complete workflow with mock S3 using: {test_file_path}")
    
    # Create components
    embedding_generator = SentenceTransformerEmbedding()
    processor = FileProcessor(embedding_generator)
    
    # Create index with correct dimension
    dimension = processor.get_index_dimension()
    index = HNSWIndex(dimension=dimension)
    
    # Create mock S3 client
    mock_s3 = MockS3Client()
    bucket_name = "test-memvec-bucket"
    
    # Run the complete workflow
    result = add_file_to_system(
        file_path=test_file_path,
        processor=processor,
        index=index,
        bucket_name=bucket_name,
        s3_client=mock_s3
    )
    
    # Verify overall success
    assert result["success"] is True, f"Workflow should succeed, got error: {result.get('error', 'None')}"
    
    # Verify basic structure
    assert "document_id" in result
    assert "total_vectors" in result
    assert "chunks_count" in result
    assert "s3_uploads" in result
    
    # Verify data
    assert result["document_id"] is not None
    assert result["total_vectors"] > 0
    assert result["chunks_count"] > 0
    assert len(result["s3_uploads"]) == result["chunks_count"]
    
    print(f"✓ Workflow completed successfully")
    print(f"  Document ID: {result['document_id']}")
    print(f"  Total vectors: {result['total_vectors']}")
    print(f"  Chunks: {result['chunks_count']}")
    
    # Verify index was populated
    assert index.size() == result["total_vectors"]
    print(f"✓ Index contains {index.size()} vectors as expected")
    
    # Verify all S3 uploads succeeded
    for i, upload in enumerate(result["s3_uploads"]):
        assert upload["success"] is True, f"Upload {i} should succeed"
        assert "chunk_id" in upload
        assert "s3_key" in upload
        assert upload["number_of_vectors"] > 0
        
        # Verify chunk ID is generated using bitwise operations
        chunk_id = upload["chunk_id"]
        assert isinstance(chunk_id, int), f"Chunk ID should be integer, got {type(chunk_id)}"
        
        # Decode chunk ID to verify it's correct
        decoded_doc, decoded_chunk = Pointer.decode_chunk_id(chunk_id)
        assert decoded_doc == result["document_id"], f"Decoded document should match: {decoded_doc} != {result['document_id']}"
        assert decoded_chunk == i, f"Decoded chunk should be {i}, got {decoded_chunk}"
        
        print(f"  Chunk {i}: {upload['number_of_vectors']} vectors, chunk_id={chunk_id} (doc={decoded_doc}, chunk={decoded_chunk})")
    
    print(f"✓ All {len(result['s3_uploads'])} chunks uploaded successfully to S3 with bitwise chunk IDs")


def test_add_file_to_system_error_handling():
    """Test workflow error handling with invalid file path."""
    
    print("Testing workflow error handling with invalid file")
    
    # Create components
    embedding_generator = SentenceTransformerEmbedding()
    processor = FileProcessor(embedding_generator)
    
    # Create index
    dimension = processor.get_index_dimension()
    index = HNSWIndex(dimension=dimension)
    
    # Create mock S3 client
    mock_s3 = MockS3Client()
    bucket_name = "test-memvec-bucket"
    
    # Try with non-existent file
    result = add_file_to_system(
        file_path="nonexistent_file.pdf",
        processor=processor,
        index=index,
        bucket_name=bucket_name,
        s3_client=mock_s3
    )
    
    # Verify failure handling
    assert result["success"] is False
    assert "error" in result
    assert result["document_id"] is None
    assert result["total_vectors"] == 0
    assert result["chunks_count"] == 0
    assert len(result["s3_uploads"]) == 0
    
    print(f"✓ Error handling works correctly")
    print(f"  Error: {result['error']}")


def test_workflow_with_chunking(test_file_path="datasets/attention.pdf"):
    """Test workflow ensures proper chunking behavior."""
    
    print(f"Testing workflow chunking behavior with: {test_file_path}")
    
    # Create components
    embedding_generator = SentenceTransformerEmbedding()
    processor = FileProcessor(embedding_generator)
    
    # Create index
    dimension = processor.get_index_dimension()
    index = HNSWIndex(dimension=dimension)
    
    # Create mock S3 client
    mock_s3 = MockS3Client()
    bucket_name = "test-memvec-bucket"
    
    # Run workflow
    result = add_file_to_system(
        file_path=test_file_path,
        processor=processor,
        index=index,
        bucket_name=bucket_name,
        s3_client=mock_s3
    )
    
    assert result["success"] is True
    
    # Verify chunk naming consistency
    document_id = result["document_id"]
    for i, upload in enumerate(result["s3_uploads"]):
        # Verify chunk ID is generated using bitwise operations
        chunk_id = upload["chunk_id"]
        decoded_doc, decoded_chunk = Pointer.decode_chunk_id(chunk_id)
        
        assert decoded_doc == document_id, f"Decoded document should match: {decoded_doc} != {document_id}"
        assert decoded_chunk == i, f"Decoded chunk should be {i}, got {decoded_chunk}"
        
        # Verify S3 key format uses the integer chunk_id
        expected_s3_key = f"chunks/{chunk_id}.pkl"
        assert upload["s3_key"] == expected_s3_key
    
    print(f"✓ Bitwise chunk ID generation and S3 key generation work correctly")
    
    # Verify total vectors equals sum of chunk vectors
    total_chunk_vectors = sum(upload["number_of_vectors"] for upload in result["s3_uploads"])
    assert total_chunk_vectors == result["total_vectors"]
    
    print(f"✓ Vector count consistency verified: {total_chunk_vectors} total vectors")
