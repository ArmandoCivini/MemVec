"""
Simple test for process_file functionality.
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processes.process_file import FileProcessor
from src.processes.components import PDFTextExtractor, SentenceTransformerEmbedding
from src.index.index import HNSWIndex


def test_process_file(test_file_path="datasets/attention.pdf"):
    """Test the complete file processing workflow using FileProcessor class."""
    
    print(f"Testing FileProcessor with: {test_file_path}")
    
    # Create configurable components
    text_extractor = PDFTextExtractor()
    embedding_generator = SentenceTransformerEmbedding()
    processor = FileProcessor(text_extractor, embedding_generator)
    
    # Create index with correct dimension using the new method
    dimension = processor.get_index_dimension()
    index = HNSWIndex(dimension=dimension)
    
    # Process the file
    vectors = processor.process_file(test_file_path, index)
    
    # Verify result is a list of vectors
    assert isinstance(vectors, list)
    assert len(vectors) > 0
    print(f"✓ File processed successfully, got {len(vectors)} vectors")
    
    # Verify all items are Vector objects
    for i, vector in enumerate(vectors):
        assert hasattr(vector, 'values'), f"Vector {i} missing values attribute"
        assert hasattr(vector, 'document'), f"Vector {i} missing document attribute"
        assert hasattr(vector, 'chunk'), f"Vector {i} missing chunk attribute"
        assert hasattr(vector, 'offset'), f"Vector {i} missing offset attribute"
        assert hasattr(vector, 'metadata'), f"Vector {i} missing metadata attribute"
        
        # Verify types
        assert isinstance(vector.values, list), f"Vector {i} values should be list"
        assert isinstance(vector.document, int), f"Vector {i} document should be int"
        assert isinstance(vector.chunk, int), f"Vector {i} chunk should be int"
        assert isinstance(vector.offset, int), f"Vector {i} offset should be int"
        assert isinstance(vector.metadata, dict), f"Vector {i} metadata should be dict"
        
        # Verify embedding dimensions - use actual dimension from processor
        expected_dimension = processor.get_index_dimension()
        assert len(vector.values) == expected_dimension, f"Vector {i} should have {expected_dimension} dimensions"
        
        # Verify metadata structure
        assert 'source_file' in vector.metadata, f"Vector {i} metadata missing source_file"
        assert 'text_index' in vector.metadata, f"Vector {i} metadata missing text_index" 
        assert 'text' in vector.metadata, f"Vector {i} metadata missing text"
    
    # Check that all vectors have same document ID 
    document_ids = [v.document for v in vectors]
    assert len(set(document_ids)) == 1, "All vectors should have same document ID"
    
    # Check chunking logic - offsets should reset when chunk changes
    chunks_and_offsets = [(v.chunk, v.offset) for v in vectors]
    
    # Verify offsets are sequential within each chunk
    current_chunk = -1
    expected_offset = 0
    for chunk, offset in chunks_and_offsets:
        if chunk != current_chunk:
            current_chunk = chunk
            expected_offset = 0
        assert offset == expected_offset, f"Offset {offset} should be {expected_offset} in chunk {chunk}"
        expected_offset += 1
    
    print(f"✓ All {len(vectors)} vectors have correct structure")
    print(f"  Document ID: {vectors[0].document}")
    print(f"  Number of chunks: {max(v.chunk for v in vectors) + 1}")
    print(f"  Sample text: {vectors[0].metadata['text'][:50]}...")
    print(f"  Embedding dimension: {dimension}")
    
    # Verify vectors were added to index
    assert index.size() == len(vectors), f"Index should contain exactly {len(vectors)} vectors"
    print(f"✓ Index contains {index.size()} vectors as expected")


def test_process_file_with_index(test_file_path="datasets/attention.pdf"):
    """Test the file processing workflow with index integration using FileProcessor class."""
    
    print(f"Testing FileProcessor with index using: {test_file_path}")
    
    # Create configurable components
    text_extractor = PDFTextExtractor()
    embedding_generator = SentenceTransformerEmbedding()
    processor = FileProcessor(text_extractor, embedding_generator)
    
    # Create index with correct dimension using the new method
    dimension = processor.get_index_dimension()
    index = HNSWIndex(dimension=dimension)
    initial_size = index.size()
    
    # Process the file with index
    vectors = processor.process_file(test_file_path, index)
    
    # Verify vectors were added to index
    final_size = index.size()
    assert final_size == initial_size + len(vectors), f"Index should contain {len(vectors)} more vectors"
    assert final_size == len(vectors), f"Index should contain exactly {len(vectors)} vectors"
    
    print(f"✓ Index now contains {final_size} vectors")
    
    # Test that we can search the index
    if len(vectors) > 0:
        query_vector = vectors[0]  # Use first vector values as query
        distances, vector_ids = index.search(query_vector, k=1)
        
        assert len(distances) == 1, "Should return exactly 1 result"
        assert len(vector_ids) == 1, "Should return exactly 1 vector ID"
        assert distances[0] < 0.001, f"Distance to identical vector should be ~0, got {distances[0]}"
        
        # The returned vector ID should match the index of the first vector
        assert vector_ids[0] == vectors[0].index, f"Returned ID {vector_ids[0]} should match vector index {vectors[0].index}"
        
        print(f"✓ Successfully searched index and found matching vector")
        print(f"  Query vector index: {vectors[0].index}")
        print(f"  Found vector ID: {vector_ids[0]}")
        print(f"  Distance: {distances[0]:.6f}")
    
    print(f"✓ Index integration test completed successfully")


def test_file_processor_class(test_file_path="datasets/attention.pdf"):
    """Test the FileProcessor class with configurable components."""
    
    print(f"Testing FileProcessor class with: {test_file_path}")
    
    # Create configurable components
    text_extractor = PDFTextExtractor()
    embedding_generator = SentenceTransformerEmbedding()
    processor = FileProcessor(text_extractor, embedding_generator)
    
    # Create index with correct dimension using the new method
    dimension = processor.get_index_dimension()
    index = HNSWIndex(dimension=dimension)
    
    # Process file using the class
    vectors = processor.process_file(test_file_path, index)
    
    # Verify results
    assert isinstance(vectors, list)
    assert len(vectors) > 0
    assert index.size() == len(vectors)
    
    print(f"✓ FileProcessor successfully processed {len(vectors)} vectors")
    print(f"✓ Embedding dimension from processor: {dimension}")
    print(f"✓ Embedding dimension from generator: {embedding_generator.dimension}")
    print(f"✓ Index contains {index.size()} vectors")
    
    # Verify the dimensions match
    assert dimension == embedding_generator.dimension, "Processor dimension should match generator dimension"



