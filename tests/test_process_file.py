"""
Simple test for process_file functionality.
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processes.process_file import process_file


def test_process_file(test_file_path="datasets/attention.pdf"):
    """Test the complete file processing workflow."""
    
    print(f"Testing process_file with: {test_file_path}")
    
    # Process the file
    vectors = process_file(test_file_path)
    
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
        
        # Verify embedding dimensions (should be 384 for all-MiniLM-L6-v2)
        assert len(vector.values) == 384, f"Vector {i} should have 384 dimensions"
        
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



