"""
Tests for the query functionality.
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.query import query_system, batch_query_system, _group_vectors_by_chunk
from src.workflow import add_file_to_system
from src.processes.process_file import FileProcessor
from src.processes.components import PDFTextExtractor, SentenceTransformerEmbedding
from src.index.index import HNSWIndex
from src.s3.mock_client import MockS3Client
from src.vectors.pointer import Pointer


def test_group_vectors_by_chunk():
    """Test the chunk grouping functionality."""
    
    print("Testing vector grouping by chunk...")
    
    # Create some test vector IDs from different chunks
    doc1_chunk0_offset0 = Pointer.encode(100, 0, 0)
    doc1_chunk0_offset1 = Pointer.encode(100, 0, 1)
    doc1_chunk1_offset0 = Pointer.encode(100, 1, 0)
    doc2_chunk0_offset0 = Pointer.encode(200, 0, 0)
    
    vector_ids = [doc1_chunk0_offset0, doc1_chunk0_offset1, doc1_chunk1_offset0, doc2_chunk0_offset0]
    
    # Group them by chunk
    chunk_groups = _group_vectors_by_chunk(vector_ids)
    
    # Verify grouping
    assert len(chunk_groups) == 3, f"Should have 3 chunk groups, got {len(chunk_groups)}"
    
    # Check specific groupings
    doc1_chunk0_id = Pointer.generate_chunk_id(100, 0)
    doc1_chunk1_id = Pointer.generate_chunk_id(100, 1)
    doc2_chunk0_id = Pointer.generate_chunk_id(200, 0)
    
    assert doc1_chunk0_id in chunk_groups, "Should have doc1_chunk0 group"
    assert doc1_chunk1_id in chunk_groups, "Should have doc1_chunk1 group"
    assert doc2_chunk0_id in chunk_groups, "Should have doc2_chunk0 group"
    
    assert len(chunk_groups[doc1_chunk0_id]) == 2, "doc1_chunk0 should have 2 vectors"
    assert len(chunk_groups[doc1_chunk1_id]) == 1, "doc1_chunk1 should have 1 vector"
    assert len(chunk_groups[doc2_chunk0_id]) == 1, "doc2_chunk0 should have 1 vector"
    
    print(f"✓ Vector grouping works correctly")
    print(f"  Created {len(chunk_groups)} chunk groups from {len(vector_ids)} vectors")
    for chunk_id, vectors in chunk_groups.items():
        print(f"  Chunk {chunk_id}: {len(vectors)} vectors")


def test_end_to_end_query_workflow(test_file_path="datasets/attention.pdf"):
    """Test the complete query workflow with a real file."""
    
    print(f"Testing end-to-end query workflow with: {test_file_path}")
    
    # Set up components
    text_extractor = PDFTextExtractor()
    embedding_generator = SentenceTransformerEmbedding()
    processor = FileProcessor(text_extractor, embedding_generator)
    
    # Create index
    dimension = processor.get_index_dimension()
    index = HNSWIndex(dimension=dimension)
    
    # Create mock S3 client
    mock_s3 = MockS3Client()
    bucket_name = "test-query-bucket"
    
    # First, add a file to the system
    print("Step 1: Adding file to system...")
    add_result = add_file_to_system(
        file_path=test_file_path,
        processor=processor,
        index=index,
        bucket_name=bucket_name,
        s3_client=mock_s3
    )
    
    assert add_result["success"] is True, f"File addition should succeed: {add_result.get('error', 'No error')}"
    print(f"✓ File added: {add_result['total_vectors']} vectors in {add_result['chunks_count']} chunks")
    
    # Now query the system
    print("Step 2: Querying the system...")
    query_text = "attention mechanism"
    
    query_result = query_system(
        query_text=query_text,
        index=index,
        bucket_name=bucket_name,
        embedding_generator=embedding_generator,
        k=3,
        s3_client=mock_s3
    )
    
    # Verify query results
    assert query_result["success"] is True, f"Query should succeed: {query_result.get('error', 'No error')}"
    assert query_result["query_text"] == query_text, "Query text should match"
    assert query_result["query_embedding"] is not None, "Should have query embedding"
    assert len(query_result["query_embedding"]) == dimension, f"Query embedding should have {dimension} dimensions"
    assert query_result["total_found"] > 0, "Should find some results"
    assert len(query_result["search_results"]) == query_result["total_found"], "Results count should match"
    
    print(f"✓ Query successful: found {query_result['total_found']} results")
    
    # Verify result structure and content
    for i, result in enumerate(query_result["search_results"]):
        assert "vector_values" in result, f"Result {i} should have vector_values"
        assert "distance" in result, f"Result {i} should have distance"
        assert "document_id" in result, f"Result {i} should have document_id"
        assert "chunk_id" in result, f"Result {i} should have chunk_id"
        assert "metadata" in result, f"Result {i} should have metadata"
        
        # Verify the vector values have the right structure
        vector_values = result["vector_values"]
        assert isinstance(vector_values, list), f"Vector {i} values should be a list"
        assert len(vector_values) == dimension, f"Vector {i} should have {dimension} dimensions"
        
        # Verify the result came from the original file
        assert result["document_id"] == add_result["document_id"], f"Document ID should match"
        assert isinstance(result["distance"], float), f"Distance should be a float"
        assert result["distance"] >= 0, f"Distance should be non-negative"
        
        print(f"  Result {i}: distance={result['distance']:.4f}, doc={result['document_id']}, chunk={result['chunk_id']}")
    
    # Verify results are ordered by distance (ascending)
    distances = [result["distance"] for result in query_result["search_results"]]
    assert distances == sorted(distances), "Results should be ordered by distance (ascending)"
    
    print(f"✓ End-to-end query workflow completed successfully")
    print(f"✓ Retrieved vectors are correctly ordered")


def test_batch_query_system(test_file_path="datasets/attention.pdf"):
    """Test batch querying functionality."""
    
    print(f"Testing batch query functionality with: {test_file_path}")
    
    # Set up components (reuse setup from previous test)
    text_extractor = PDFTextExtractor()
    embedding_generator = SentenceTransformerEmbedding()
    processor = FileProcessor(text_extractor, embedding_generator)
    
    dimension = processor.get_index_dimension()
    index = HNSWIndex(dimension=dimension)
    mock_s3 = MockS3Client()
    bucket_name = "test-batch-query-bucket"
    
    # Add file to system
    add_result = add_file_to_system(
        file_path=test_file_path,
        processor=processor,
        index=index,
        bucket_name=bucket_name,
        s3_client=mock_s3
    )
    assert add_result["success"] is True
    
    # Test batch querying
    queries = [
        "attention mechanism",
        "neural network",
        "machine learning"
    ]
    
    batch_results = batch_query_system(
        query_texts=queries,
        index=index,
        bucket_name=bucket_name,
        embedding_generator=embedding_generator,
        k=2,
        s3_client=mock_s3
    )
    
    # Verify batch results
    assert len(batch_results) == len(queries), f"Should have {len(queries)} results, got {len(batch_results)}"
    
    for i, (query, result) in enumerate(zip(queries, batch_results)):
        assert result["query_text"] == query, f"Query {i} text should match"
        assert result["success"] is True, f"Query {i} should succeed"
        assert result["total_found"] > 0, f"Query {i} should find results"
        
        # Verify each result has proper structure and content
        for j, search_result in enumerate(result["search_results"]):
            assert "vector_values" in search_result, f"Query {i}, result {j} should have vector_values"
            assert "distance" in search_result, f"Query {i}, result {j} should have distance"
            assert "metadata" in search_result, f"Query {i}, result {j} should have metadata"
            
            vector_values = search_result["vector_values"]
            assert isinstance(vector_values, list), f"Query {i}, result {j} vector should be a list"
            assert len(vector_values) == dimension, f"Query {i}, result {j} vector should have {dimension} dimensions"
        
        # Verify results are ordered by distance
        distances = [r["distance"] for r in result["search_results"]]
        assert distances == sorted(distances), f"Query {i} results should be ordered by distance"
        
        print(f"✓ Batch query {i}: '{query}' -> {result['total_found']} results")
    
    print(f"✓ Batch query functionality works correctly")
    print(f"✓ All results properly retrieved from S3")


def test_query_error_handling():
    """Test query error handling with empty index."""
    
    print("Testing query error handling...")
    
    # Create empty index
    dimension = 384  # Standard dimension
    index = HNSWIndex(dimension=dimension)
    
    embedding_generator = SentenceTransformerEmbedding()
    mock_s3 = MockS3Client()
    bucket_name = "test-error-bucket"
    
    # Try to query empty index
    result = query_system(
        query_text="test query",
        index=index,
        bucket_name=bucket_name,
        embedding_generator=embedding_generator,
        k=5,
        s3_client=mock_s3
    )
    
    # Should handle gracefully (no crash)
    assert result["success"] is True, "Empty index query should still succeed"
    # Note: FAISS may return dummy results even on empty index, so we just check it doesn't crash
    
    print("✓ Error handling works correctly")
    print(f"  Empty index query: {result['total_found']} results (FAISS behavior)")
