"""
Test Vector class integration with Pointer.
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectors.vectors import Vector
from src.vectors.pointer import Pointer


def test_vector_with_pointer():
    """Test Vector creation with pointer components."""
    
    # Create vector with pointer components
    vec = Vector(
        values=[1.0, 2.0, 3.0],
        document=100,
        chunk=50,
        offset=25
    )
    
    # Verify pointer integration
    assert vec.index is not None
    assert vec.document == 100
    assert vec.chunk == 50
    assert vec.offset == 25
    
    # Verify the index matches manual encoding
    expected_index = Pointer.encode(100, 50, 25)
    assert vec.index == expected_index
    
    print(f"✓ Vector with pointer created: {vec}")
    
    return True


def test_get_pointer_components():
    """Test getting pointer components from vector."""
    
    vec = Vector(
        values=[7.0, 8.0, 9.0],
        document=200,
        chunk=75,
        offset=10
    )
    
    components = vec.get_pointer_components()
    assert components == (200, 75, 10)
    
    print(f"✓ Retrieved pointer components: {components}")
    
    return True


def test_set_pointer_components():
    """Test setting pointer components on existing vector."""
    
    # Create vector with initial pointer
    vec = Vector(values=[1.0, 2.0, 3.0], document=1, chunk=1, offset=1)
    
    # Set new pointer components
    vec.set_pointer_components(300, 100, 5)
    
    # Verify update
    assert vec.document == 300
    assert vec.chunk == 100
    assert vec.offset == 5
    assert vec.index == Pointer.encode(300, 100, 5)
    
    print(f"✓ Updated vector with pointer: {vec}")
    
    return True


def test_index_decode_consistency():
    """Test that encoding and decoding are consistent."""
    
    test_cases = [
        (1, 2, 3),
        (1000, 500, 100),
        (0, 0, 0)
    ]
    
    for doc, chunk, offset in test_cases:
        vec = Vector(
            values=[1.0, 2.0],
            document=doc,
            chunk=chunk,
            offset=offset
        )
        
        # Get components back
        decoded = vec.get_pointer_components()
        assert decoded == (doc, chunk, offset)
        
        print(f"✓ Encode/decode consistency for ({doc}, {chunk}, {offset})")
    
    return True


def test_vector_with_unified_index():
    """Test Vector creation with raw integer index."""
    
    # Create a raw integer index manually
    doc, chunk, offset = 150, 75, 30
    raw_index = Pointer.encode(doc, chunk, offset)
    
    # Create vector using raw integer index (like what FAISS search would return)
    vec = Vector(
        values=[4.0, 5.0, 6.0],
        index=raw_index  # Raw integer, not Pointer object
    )
    
    # Verify components were correctly decoded
    assert vec.index == raw_index
    assert vec.document == doc
    assert vec.chunk == chunk
    assert vec.offset == offset
    
    # Verify consistency with get_pointer_components
    decoded_components = vec.get_pointer_components()
    assert decoded_components == (doc, chunk, offset)
    
    print(f"✓ Vector with raw integer index created: {vec}")
    print(f"✓ Raw index: {raw_index}")
    print(f"✓ Components correctly decoded: doc={vec.document}, chunk={vec.chunk}, offset={vec.offset}")
    
    return True


if __name__ == "__main__":
    print("Vector-Pointer Integration Tests")
    print("=" * 35)
    
    try:
        test_results = []
        
        print("\n1. Testing vector with pointer...")
        test_results.append(test_vector_with_pointer())
        
        print("\n2. Testing get pointer components...")
        test_results.append(test_get_pointer_components())
        
        print("\n3. Testing set pointer components...")
        test_results.append(test_set_pointer_components())
        
        print("\n4. Testing encode/decode consistency...")
        test_results.append(test_index_decode_consistency())
        
        print("\n5. Testing vector with unified index...")
        test_results.append(test_vector_with_unified_index())
        
        if all(test_results):
            print("\n✓ All integration tests passed!")
        else:
            print("\n✗ Some tests failed!")
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
