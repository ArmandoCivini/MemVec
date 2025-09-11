"""
Simple test for process_file functionality.
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processes.process_file import FileProcessor
from src.processes.components import SentenceTransformerEmbedding
from src.index.index import HNSWIndex


def test_process_file(test_file_path="datasets/attention.pdf"):
    """Test the complete file processing workflow using FileProcessor class."""
    
    print(f"Testing FileProcessor with: {test_file_path}")
    
    # Create configurable components
    embedding_generator = SentenceTransformerEmbedding()
    processor = FileProcessor(embedding_generator)
    
    # Create index with correct dimension using the new method
    dimension = processor.get_index_dimension()
    index = HNSWIndex(dimension=dimension)
    
    # Process the file using file object
    filename = os.path.basename(test_file_path)
    with open(test_file_path, 'rb') as file_obj:
        chunks = processor.process_file(file_obj, filename, index)
    
    # Verify result is a list of chunks (lists of vectors)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    
    # Flatten chunks to get all vectors for testing
    all_vectors = [vector for chunk in chunks for vector in chunk]
    
    print(f"✓ File processed successfully, got {len(chunks)} chunks with {len(all_vectors)} total vectors")
    
    # Verify all chunks contain Vector objects
    for chunk_idx, chunk in enumerate(chunks):
        assert isinstance(chunk, list), f"Chunk {chunk_idx} should be a list"
        assert len(chunk) > 0, f"Chunk {chunk_idx} should not be empty"
        
        for i, vector in enumerate(chunk):
            assert hasattr(vector, 'values'), f"Vector {i} in chunk {chunk_idx} missing values attribute"
            assert hasattr(vector, 'document'), f"Vector {i} in chunk {chunk_idx} missing document attribute"
            assert hasattr(vector, 'chunk'), f"Vector {i} in chunk {chunk_idx} missing chunk attribute"
            assert hasattr(vector, 'offset'), f"Vector {i} in chunk {chunk_idx} missing offset attribute"
            assert hasattr(vector, 'metadata'), f"Vector {i} in chunk {chunk_idx} missing metadata attribute"
            
            # Verify types
            assert isinstance(vector.values, list), f"Vector {i} in chunk {chunk_idx} values should be list"
            assert isinstance(vector.document, int), f"Vector {i} in chunk {chunk_idx} document should be int"
            assert isinstance(vector.chunk, int), f"Vector {i} in chunk {chunk_idx} chunk should be int"
            assert isinstance(vector.offset, int), f"Vector {i} in chunk {chunk_idx} offset should be int"
            assert isinstance(vector.metadata, dict), f"Vector {i} in chunk {chunk_idx} metadata should be dict"
            
            # Verify embedding dimensions - use actual dimension from processor
            expected_dimension = processor.get_index_dimension()
            assert len(vector.values) == expected_dimension, f"Vector {i} in chunk {chunk_idx} should have {expected_dimension} dimensions"
            
            # Verify metadata structure
            assert 'source_file' in vector.metadata, f"Vector {i} in chunk {chunk_idx} metadata missing source_file"
            assert 'text_index' in vector.metadata, f"Vector {i} in chunk {chunk_idx} metadata missing text_index" 
            assert 'text' in vector.metadata, f"Vector {i} in chunk {chunk_idx} metadata missing text"
    
    # Check that all vectors have same document ID 
    document_ids = [v.document for v in all_vectors]
    assert len(set(document_ids)) == 1, "All vectors should have same document ID"
    
    # Check chunking logic - each chunk should have sequential offsets starting from 0
    for chunk_idx, chunk in enumerate(chunks):
        for i, vector in enumerate(chunk):
            assert vector.chunk == chunk_idx, f"Vector in chunk {chunk_idx} should have chunk number {chunk_idx}"
            assert vector.offset == i, f"Vector at position {i} in chunk {chunk_idx} should have offset {i}"
    
    print(f"✓ All {len(all_vectors)} vectors in {len(chunks)} chunks have correct structure")
    print(f"  Document ID: {all_vectors[0].document}")
    print(f"  Number of chunks: {len(chunks)}")
    print(f"  Sample text: {all_vectors[0].metadata['text'][:50]}...")
    print(f"  Embedding dimension: {dimension}")
    
    # Verify vectors were added to index
    assert index.size() == len(all_vectors), f"Index should contain exactly {len(all_vectors)} vectors"
    print(f"✓ Index contains {index.size()} vectors as expected")


def test_process_file_with_index(test_file_path="datasets/attention.pdf"):
    """Test the file processing workflow with index integration using FileProcessor class."""
    
    print(f"Testing FileProcessor with index using: {test_file_path}")
    
    # Create configurable components
    embedding_generator = SentenceTransformerEmbedding()
    processor = FileProcessor(embedding_generator)
    
    # Create index with correct dimension using the new method
    dimension = processor.get_index_dimension()
    index = HNSWIndex(dimension=dimension)
    initial_size = index.size()
    
    # Process the file with index using file object
    filename = os.path.basename(test_file_path)
    with open(test_file_path, 'rb') as file_obj:
        chunks = processor.process_file(file_obj, filename, index)
    
    # Flatten chunks to get all vectors
    all_vectors = [vector for chunk in chunks for vector in chunk]
    
    # Verify vectors were added to index
    final_size = index.size()
    assert final_size == initial_size + len(all_vectors), f"Index should contain {len(all_vectors)} more vectors"
    assert final_size == len(all_vectors), f"Index should contain exactly {len(all_vectors)} vectors"
    
    print(f"✓ Index now contains {final_size} vectors from {len(chunks)} chunks")
    
    # Test that we can search the index
    if len(all_vectors) > 0:
        query_vector = all_vectors[0]  # Use first vector values as query
        distances, vector_ids = index.search(query_vector, k=1)
        
        assert len(distances) == 1, "Should return exactly 1 result"
        assert len(vector_ids) == 1, "Should return exactly 1 vector ID"
        assert distances[0] < 0.001, f"Distance to identical vector should be ~0, got {distances[0]}"
        
        # The returned vector ID should match the index of the first vector
        assert vector_ids[0] == all_vectors[0].index, f"Returned ID {vector_ids[0]} should match vector index {all_vectors[0].index}"
        
        print(f"✓ Successfully searched index and found matching vector")
        print(f"  Query vector index: {all_vectors[0].index}")
        print(f"  Found vector ID: {vector_ids[0]}")
        print(f"  Distance: {distances[0]:.6f}")
    
    print(f"✓ Index integration test completed successfully")


def test_file_processor_class(test_file_path="datasets/attention.pdf"):
    """Test the FileProcessor class with configurable components."""
    
    print(f"Testing FileProcessor class with: {test_file_path}")
    
    # Create configurable components
    embedding_generator = SentenceTransformerEmbedding()
    processor = FileProcessor(embedding_generator)
    
    # Create index with correct dimension using the new method
    dimension = processor.get_index_dimension()
    index = HNSWIndex(dimension=dimension)
    
    # Process file using the class with file object
    filename = os.path.basename(test_file_path)
    with open(test_file_path, 'rb') as file_obj:
        chunks = processor.process_file(file_obj, filename, index)
    
    # Flatten chunks to get all vectors
    all_vectors = [vector for chunk in chunks for vector in chunk]
    
    # Verify results
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert index.size() == len(all_vectors)
    
    print(f"✓ FileProcessor successfully processed {len(all_vectors)} vectors in {len(chunks)} chunks")
    print(f"✓ Embedding dimension from processor: {dimension}")
    print(f"✓ Embedding dimension from generator: {embedding_generator.dimension}")
    print(f"✓ Index contains {index.size()} vectors")
    
    # Verify the dimensions match
    assert dimension == embedding_generator.dimension, "Processor dimension should match generator dimension"



