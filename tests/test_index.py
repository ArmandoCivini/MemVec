"""
Simple tests for FAISS HNSW index functionality.
"""

import sys
import os
import numpy as np

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.index.index import HNSWIndex
from src.vectors.vectors import Vector


def test_index_operations():
    """Test HNSW index creation, addition, and search operations."""
    
    # Create test Vector objects with 3-dimensional embeddings
    test_vectors = [
        Vector(id="vec-1", values=[1.0, 0.0, 0.0]),
        Vector(id="vec-2", values=[0.0, 1.0, 0.0]),
        Vector(id="vec-3", values=[0.0, 0.0, 1.0]),
        Vector(id="vec-4", values=[1.0, 1.0, 0.0]),
        Vector(id="vec-5", values=[0.5, 0.5, 0.5])
    ]
    
    # Create HNSW index
    dimension = 3
    index = HNSWIndex(dimension=dimension, m=16)
    print(f"✓ Created HNSW index with dimension {dimension}")
    
    # Test initial state
    info = index.get_info()
    assert info["total_vectors"] == 0
    assert info["dimension"] == dimension
    assert info["m"] == 16
    assert info["is_trained"] == True  # HNSW doesn't require training
    assert index.size() == 0
    print("✓ Initial state verified")
    
    # Add vectors to index
    index.add_vectors(test_vectors)
    print(f"✓ Added {len(test_vectors)} vectors to index")
    
    # Test state after adding vectors
    assert index.size() == 5
    info = index.get_info()
    assert info["total_vectors"] == 5
    print("✓ Vector count verified after addition")
    
    # Test search with query vector similar to vec-1
    query_vector = Vector(id="query", values=[0.9, 0.1, 0.1])
    distances, indices = index.search(query_vector, k=3)
    
    # Check search results
    assert len(distances) == 3
    assert len(indices) == 3
    assert indices[0] == 0  # Should find vec-1 (index 0) as closest
    print(f"✓ Search found closest vector at index {indices[0]} with distance {distances[0]:.4f}")
    
    # Test search with different k value
    distances_k5, indices_k5 = index.search(query_vector, k=5)
    assert len(distances_k5) == 5
    assert len(indices_k5) == 5
    print("✓ Search with k=5 returned correct number of results")
    
    # Test search with vector in the middle
    middle_query = Vector(id="middle-query", values=[0.3, 0.3, 0.3])
    distances_middle, indices_middle = index.search(middle_query, k=2)
    assert len(distances_middle) == 2
    print(f"✓ Middle search found vectors at indices {indices_middle}")
    
    return True


def test_empty_vector_addition():
    """Test adding empty vector list."""
    
    index = HNSWIndex(dimension=3)
    
    # Test adding empty list
    index.add_vectors([])
    assert index.size() == 0
    print("✓ Empty vector addition handled correctly")
    
    return True


def test_single_vector_operations():
    """Test operations with a single vector."""
    
    index = HNSWIndex(dimension=2)
    
    # Add single vector
    single_vector = [Vector(id="single", values=[1.0, 2.0])]
    index.add_vectors(single_vector)
    
    assert index.size() == 1
    print("✓ Single vector addition successful")
    
    # Search with single vector in index
    query = Vector(id="query", values=[1.1, 2.1])
    distances, indices = index.search(query, k=1)
    
    assert len(distances) == 1
    assert len(indices) == 1
    assert indices[0] == 0
    print(f"✓ Single vector search successful, distance: {distances[0]:.4f}")
    
    return True


def test_multi_search():
    """Test multi-search functionality with multiple query vectors."""
    
    # Create index with test vectors
    index = HNSWIndex(dimension=3)
    
    test_vectors = [
        Vector(id="vec-1", values=[1.0, 0.0, 0.0]),
        Vector(id="vec-2", values=[0.0, 1.0, 0.0]),
        Vector(id="vec-3", values=[0.0, 0.0, 1.0]),
        Vector(id="vec-4", values=[1.0, 1.0, 0.0])
    ]
    
    index.add_vectors(test_vectors)
    print("✓ Added vectors for multi-search test")
    
    # Create multiple query vectors
    query_vectors = [
        Vector(id="query-1", values=[0.9, 0.1, 0.0]),  # Similar to vec-1
        Vector(id="query-2", values=[0.1, 0.9, 0.0]),  # Similar to vec-2
        Vector(id="query-3", values=[0.5, 0.5, 0.0])   # Similar to vec-4
    ]
    
    # Perform multi-search
    distances, indices = index.multi_search(query_vectors, k=2)
    
    # Check results structure
    assert len(distances) == 3  # 3 query vectors
    assert len(indices) == 3    # 3 query vectors
    
    for i, (dist_list, idx_list) in enumerate(zip(distances, indices)):
        assert len(dist_list) == 2  # k=2
        assert len(idx_list) == 2   # k=2
        print(f"✓ Query {i+1} found vectors at indices {idx_list} with distances {[f'{d:.4f}' for d in dist_list]}")
    
    # Test with empty query list
    empty_distances, empty_indices = index.multi_search([], k=2)
    assert empty_distances == []
    assert empty_indices == []
    print("✓ Empty multi-search handled correctly")
    
    return True


if __name__ == "__main__":
    print("HNSW Index Tests")
    print("=" * 20)
    
    try:
        # Run all tests
        test_results = []
        
        print("\n1. Testing index operations...")
        test_results.append(test_index_operations())
        
        print("\n2. Testing empty vector addition...")
        test_results.append(test_empty_vector_addition())
        
        print("\n3. Testing single vector operations...")
        test_results.append(test_single_vector_operations())
        
        print("\n4. Testing multi-search operations...")
        test_results.append(test_multi_search())
        
        # Check all tests passed
        if all(test_results):
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Some tests failed!")
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
