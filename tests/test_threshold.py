"""
Test threshold functionality in the MemVec index.
"""

import pytest
from src.index.index import HNSWIndex
from src.vectors.vectors import Vector


def test_threshold_filtering_basic():
    """Test that threshold filtering works at the index level."""
    
    # Create index
    index = HNSWIndex(dimension=3)
    
    # Add some test vectors with known distances from query point [1.0, 0.0, 0.0]
    # Note: FAISS uses squared L2 distance
    test_vectors = [
        Vector(values=[1.0, 0.0, 0.0], index=100),  # Squared distance = 0.0 
        Vector(values=[0.9, 0.1, 0.0], index=101),  # Squared distance ≈ 0.02
        Vector(values=[0.5, 0.5, 0.0], index=102),  # Squared distance = 0.5 
        Vector(values=[0.0, 1.0, 0.0], index=103),  # Squared distance = 2.0 
    ]
    
    index.add_vectors(test_vectors)
    
    # Query vector
    query = [1.0, 0.0, 0.0]
    k = 10  # Ask for more than available
    
    # Search without threshold
    distances_no_threshold, ids_no_threshold = index.search(query, k=k)
    
    # Search with threshold = 0.51 (should include vector with distance 0.5)
    # Note: FAISS range_search uses strict inequality (<), not (<=)
    threshold = 0.51
    distances_with_threshold, ids_with_threshold = index.search(query, k=k, threshold=threshold)
    
    # Verify threshold filtering worked
    assert len(distances_with_threshold) < len(distances_no_threshold), "Threshold should reduce results"
    assert all(d < threshold for d in distances_with_threshold), f"All distances should be < {threshold}"
    assert len(distances_with_threshold) == 3, "Should have exactly 3 results within threshold 0.51"
    
    print(f"✓ Without threshold: {len(distances_no_threshold)} results")
    print(f"✓ With threshold {threshold}: {len(distances_with_threshold)} results")


def test_threshold_filtering_strict():
    """Test threshold filtering with a very strict threshold."""
    
    index = HNSWIndex(dimension=2)
    
    # Add vectors with varying distances
    test_vectors = [
        Vector(values=[1.0, 0.0], index=200),  # Distance ~0.0 from [1.0, 0.0]
        Vector(values=[0.8, 0.0], index=201),  # Distance ~0.2 from [1.0, 0.0]
        Vector(values=[0.5, 0.0], index=202),  # Distance ~0.5 from [1.0, 0.0]
    ]
    
    index.add_vectors(test_vectors)
    
    query = [1.0, 0.0]
    
    # Very strict threshold should only return the exact match
    strict_threshold = 0.1
    distances, ids = index.search(query, k=5, threshold=strict_threshold)
    
    assert len(distances) <= 2, "Should have at most 2 results with strict threshold"
    assert all(d <= strict_threshold for d in distances), f"All distances should be <= {strict_threshold}"
    assert 200 in ids, "Should include the exact match"
    
    print(f"✓ Strict threshold {strict_threshold}: {len(distances)} results")


def test_threshold_filtering_none():
    """Test that None threshold behaves like no threshold."""
    
    index = HNSWIndex(dimension=2)
    
    test_vectors = [
        Vector(values=[1.0, 0.0], index=300),
        Vector(values=[0.0, 1.0], index=301),
        Vector(values=[-1.0, 0.0], index=302),
    ]
    
    index.add_vectors(test_vectors)
    
    query = [1.0, 0.0]
    k = 5
    
    # Search with None threshold should be same as no threshold
    distances_none, ids_none = index.search(query, k=k, threshold=None)
    distances_no_param, ids_no_param = index.search(query, k=k)
    
    assert distances_none == distances_no_param, "None threshold should behave like no threshold"
    assert ids_none == ids_no_param, "None threshold should return same IDs as no threshold"
    
    print(f"✓ None threshold behaves correctly")


def test_threshold_multi_search():
    """Test threshold filtering with multi-search functionality."""
    
    index = HNSWIndex(dimension=2)
    
    test_vectors = [
        Vector(values=[1.0, 0.0], index=400),
        Vector(values=[0.0, 1.0], index=401),
        Vector(values=[0.9, 0.1], index=402),
        Vector(values=[0.1, 0.9], index=403),
    ]
    
    index.add_vectors(test_vectors)
    
    # Multiple query vectors
    query_vectors = [
        Vector(values=[1.0, 0.0], document=0, chunk=0, offset=0),
        Vector(values=[0.0, 1.0], document=0, chunk=0, offset=1),
    ]
    
    k = 3
    threshold = 0.5
    
    # Multi-search with threshold
    distances_list, ids_list = index.multi_search(query_vectors, k=k, threshold=threshold)
    
    assert len(distances_list) == 2, "Should return results for both queries"
    assert len(ids_list) == 2, "Should return IDs for both queries"
    
    # Check that all distances are within threshold
    for distances in distances_list:
        assert all(d <= threshold for d in distances), f"All distances should be <= {threshold}"
    
    print(f"✓ Multi-search with threshold works correctly")


def test_threshold_empty_results():
    """Test threshold that returns no results."""
    
    index = HNSWIndex(dimension=2)
    
    # Add vectors that are far from query
    test_vectors = [
        Vector(values=[10.0, 10.0], index=500),
        Vector(values=[-10.0, -10.0], index=501),
    ]
    
    index.add_vectors(test_vectors)
    
    query = [0.0, 0.0]
    very_strict_threshold = 0.1
    
    distances, ids = index.search(query, k=5, threshold=very_strict_threshold)
    
    assert len(distances) == 0, "Should return no results with very strict threshold"
    assert len(ids) == 0, "Should return no IDs with very strict threshold"
    
    print(f"✓ Very strict threshold correctly returns empty results")
