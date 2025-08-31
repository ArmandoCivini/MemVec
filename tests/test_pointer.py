"""
Simple tests for Pointer class functionality.
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectors.pointer import Pointer, generate_document_id
from src.config.contants import OFFSET_BITS, CHUNK_BITS, DOCUMENT_BITS


def test_encode_decode():
    """Test basic encode/decode operations."""
    
    # Test basic encoding/decoding
    test_cases = [
        (0, 0, 0),
        (1, 2, 3),
        (100, 500, 1000),
        (Pointer.MAX_DOCUMENT, Pointer.MAX_CHUNK, Pointer.MAX_OFFSET)
    ]
    
    for doc, chunk, offset in test_cases:
        # Encode
        index = Pointer.encode(doc, chunk, offset)
        
        # Decode
        decoded_doc, decoded_chunk, decoded_offset = Pointer.decode(index)
        
        # Verify
        assert decoded_doc == doc, f"Document mismatch: {decoded_doc} != {doc}"
        assert decoded_chunk == chunk, f"Chunk mismatch: {decoded_chunk} != {chunk}"
        assert decoded_offset == offset, f"Offset mismatch: {decoded_offset} != {offset}"
        
        print(f"✓ Encode/decode test passed for ({doc}, {chunk}, {offset}) -> {index}")
    
    return True


def test_limits():
    """Test the document ID generator."""
    
    # Test document ID generation
    for _ in range(10):
        doc_id = generate_document_id()
        assert 0 <= doc_id <= Pointer.MAX_DOCUMENT, f"Generated doc_id {doc_id} out of range"
    
    print(f"✓ Document ID generator produces valid values (0 to {Pointer.MAX_DOCUMENT})")
    
    return True


def test_get_limits():
    """Test the get_limits method."""
    
    limits = Pointer.get_limits()
    
    assert "max_document" in limits
    assert "max_chunk" in limits
    assert "max_offset" in limits
    assert "total_capacity" in limits
    
    print(f"✓ Limits: {limits}")
    print(f"✓ Total theoretical capacity: {limits['total_capacity']:,} vectors")
    
    return True


def test_bit_allocation():
    """Test that bit allocation adds up correctly."""
    
    total_bits = OFFSET_BITS + CHUNK_BITS + DOCUMENT_BITS
    print(f"✓ Total bits used: {total_bits}/64")
    
    assert total_bits <= 64, f"Total bits {total_bits} exceeds 64-bit integer limit"
    
    # Test that each component uses its allocated bits correctly
    max_values = {
        "offset": 2**OFFSET_BITS - 1,
        "chunk": 2**CHUNK_BITS - 1,
        "document": 2**DOCUMENT_BITS - 1
    }
    
    assert max_values["offset"] == Pointer.MAX_OFFSET
    assert max_values["chunk"] == Pointer.MAX_CHUNK
    assert max_values["document"] == Pointer.MAX_DOCUMENT
    
    print(f"✓ Bit allocation verified: {DOCUMENT_BITS}+{CHUNK_BITS}+{OFFSET_BITS} = {total_bits} bits")
    
    return True


if __name__ == "__main__":
    print("Pointer Class Tests")
    print("=" * 20)
    
    try:
        test_results = []
        
        print("\n1. Testing encode/decode operations...")
        test_results.append(test_encode_decode())
        
        print("\n2. Testing document ID generator...")
        test_results.append(test_limits())
        
        print("\n3. Testing get_limits method...")
        test_results.append(test_get_limits())
        
        print("\n4. Testing bit allocation...")
        test_results.append(test_bit_allocation())
        
        if all(test_results):
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Some tests failed!")
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
