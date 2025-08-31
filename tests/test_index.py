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
    
    # Create test Vector objects with 3-dimensional embeddings and explicit indices
    test_vectors = [
        Vector(values=[1.0, 0.0, 0.0], index=100),
        Vector(values=[0.0, 1.0, 0.0], index=200),
        Vector(values=[0.0, 0.0, 1.0], index=300),
        Vector(values=[1.0, 1.0, 0.0], index=400),
        Vector(values=[0.5, 0.5, 0.5], index=500)
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
    query_vector = Vector(values=[0.9, 0.1, 0.1], document=0, chunk=0, offset=0)
    distances, vector_ids = index.search(query_vector, k=3)
    
    # Check search results
    assert len(distances) == 3
    assert len(vector_ids) == 3
    assert vector_ids[0] == 100  # Should find vec-1 (index 100) as closest
    print(f"✓ Search found closest vector with ID {vector_ids[0]} at distance {distances[0]:.4f}")
    
    # Test search with different k value
    distances_k5, vector_ids_k5 = index.search(query_vector, k=5)
    assert len(distances_k5) == 5
    assert len(vector_ids_k5) == 5
    print("✓ Search with k=5 returned correct number of results")
    
    # Test search with vector in the middle
    middle_query = Vector(values=[0.3, 0.3, 0.3], document=0, chunk=0, offset=0)
    distances_middle, vector_ids_middle = index.search(middle_query, k=2)
    assert len(distances_middle) == 2
    print(f"✓ Middle search found vectors with IDs {vector_ids_middle}")


def test_empty_vector_addition():
    """Test adding empty vector list."""
    
    index = HNSWIndex(dimension=3)
    
    # Test adding empty list
    index.add_vectors([])
    assert index.size() == 0
    print("✓ Empty vector addition handled correctly")


def test_single_vector_operations():
    """Test operations with a single vector."""
    
    index = HNSWIndex(dimension=2)
    
    # Add single vector with explicit index
    single_vector = [Vector(values=[1.0, 2.0], index=999)]
    index.add_vectors(single_vector)
    
    assert index.size() == 1
    print("✓ Single vector addition successful")
    
    # Search with single vector in index
    query = Vector(values=[1.1, 2.1], document=0, chunk=0, offset=0)
    distances, vector_ids = index.search(query, k=1)
    
    assert len(distances) == 1
    assert len(vector_ids) == 1
    assert vector_ids[0] == 999
    print(f"✓ Single vector search successful, found vector ID {vector_ids[0]} at distance {distances[0]:.4f}")


def test_multi_search():
    """Test multi-search functionality with multiple query vectors."""
    
    # Create index with test vectors
    index = HNSWIndex(dimension=3)
    
    test_vectors = [
        Vector(values=[1.0, 0.0, 0.0], index=101),
        Vector(values=[0.0, 1.0, 0.0], index=102),
        Vector(values=[0.0, 0.0, 1.0], index=103),
        Vector(values=[1.0, 1.0, 0.0], index=104)
    ]
    
    index.add_vectors(test_vectors)
    print("✓ Added vectors for multi-search test")
    
    # Create multiple query vectors
    query_vectors = [
        Vector(values=[0.9, 0.1, 0.0], document=0, chunk=0, offset=0),  # Similar to vec-1
        Vector(values=[0.1, 0.9, 0.0], document=0, chunk=0, offset=1),  # Similar to vec-2
        Vector(values=[0.5, 0.5, 0.0], document=0, chunk=0, offset=2)   # Similar to vec-4
    ]
    
    # Perform multi-search
    distances, vector_ids = index.multi_search(query_vectors, k=2)
    
    # Check results structure
    assert len(distances) == 3  # 3 query vectors
    assert len(vector_ids) == 3    # 3 query vectors
    
    for i, (dist_list, id_list) in enumerate(zip(distances, vector_ids)):
        assert len(dist_list) == 2  # k=2
        assert len(id_list) == 2   # k=2
        print(f"✓ Query {i+1} found vectors with IDs {id_list} at distances {[f'{d:.4f}' for d in dist_list]}")
    
    # Test with empty query list
    empty_distances, empty_vector_ids = index.multi_search([], k=2)
    assert empty_distances == []
    assert empty_vector_ids == []
    print("✓ Empty multi-search handled correctly")


def test_vectors_without_indices():
    """Test adding vectors without explicit indices (fallback to hash)."""
    
    index = HNSWIndex(dimension=2)
    
    # Create vectors without explicit indices
    test_vectors = [
        Vector(values=[1.0, 0.0], document=1, chunk=0, offset=0),
        Vector(values=[0.0, 1.0], document=1, chunk=0, offset=1)
    ]
    
    # Add vectors (should use hash fallback)
    index.add_vectors(test_vectors)
    assert index.size() == 2
    print("✓ Vectors with pointer components added successfully")
    
    # Search should still work
    query = Vector(values=[0.9, 0.1], document=0, chunk=0, offset=0)
    distances, vector_ids = index.search(query, k=1)
    
    assert len(distances) == 1
    assert len(vector_ids) == 1
    print(f"✓ Search with pointer-based indices successful, found ID {vector_ids[0]}")


def test_empty_vector_addition():
    """Test adding empty vector list."""
    
    index = HNSWIndex(dimension=3)
    
    # Test adding empty list
    index.add_vectors([])
    assert index.size() == 0
    print("✓ Empty vector addition handled correctly")


def test_single_vector_operations():
    """Test operations with a single vector."""
    
    index = HNSWIndex(dimension=2)
    
    # Add single vector with explicit index
    single_vector = [Vector(values=[1.0, 2.0], index=999)]
    index.add_vectors(single_vector)
    
    assert index.size() == 1
    print("✓ Single vector addition successful")
    
    # Search with single vector in index
    query = Vector(values=[1.1, 2.1], document=0, chunk=0, offset=0)
    distances, vector_ids = index.search(query, k=1)
    
    assert len(distances) == 1
    assert len(vector_ids) == 1
    assert vector_ids[0] == 999
    print(f"✓ Single vector search successful, found vector ID {vector_ids[0]} at distance {distances[0]:.4f}")


def test_multi_search():
    """Test multi-search functionality with multiple query vectors."""
    
    # Create index with test vectors
    index = HNSWIndex(dimension=3)
    
    test_vectors = [
        Vector(values=[1.0, 0.0, 0.0], index=101),
        Vector(values=[0.0, 1.0, 0.0], index=102),
        Vector(values=[0.0, 0.0, 1.0], index=103),
        Vector(values=[1.0, 1.0, 0.0], index=104)
    ]
    
    index.add_vectors(test_vectors)
    print("✓ Added vectors for multi-search test")
    
    # Create multiple query vectors
    query_vectors = [
        Vector(values=[0.9, 0.1, 0.0], document=0, chunk=0, offset=0),  # Similar to vec-1
        Vector(values=[0.1, 0.9, 0.0], document=0, chunk=0, offset=1),  # Similar to vec-2
        Vector(values=[0.5, 0.5, 0.0], document=0, chunk=0, offset=2)   # Similar to vec-4
    ]
    
    # Perform multi-search
    distances, vector_ids = index.multi_search(query_vectors, k=2)
    
    # Check results structure
    assert len(distances) == 3  # 3 query vectors
    assert len(vector_ids) == 3    # 3 query vectors
    
    for i, (dist_list, id_list) in enumerate(zip(distances, vector_ids)):
        assert len(dist_list) == 2  # k=2
        assert len(id_list) == 2   # k=2
        print(f"✓ Query {i+1} found vectors with IDs {id_list} at distances {[f'{d:.4f}' for d in dist_list]}")
    
    # Test with empty query list
    empty_distances, empty_vector_ids = index.multi_search([], k=2)
    assert empty_distances == []
    assert empty_vector_ids == []
    print("✓ Empty multi-search handled correctly")




