"""
Simple tests for CacheLayer functionality.
"""

import sys
import os
import numpy as np

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache.cache_layer import CacheLayer


def test_cache_basic_operations():
    """Test basic set/get operations."""
    
    cache = CacheLayer(use_fake=True)
    cache.clear()  # Start with clean cache
    
    # Test string value
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"
    
    # Test numpy array
    vector = np.array([1.0, 2.0, 3.0])
    cache.set("vector", vector)
    retrieved = cache.get("vector")
    assert np.array_equal(vector, retrieved)
    
    # Test non-existent key
    assert cache.get("missing") is None
    
    # Test exists
    assert cache.exists("key1") is True
    assert cache.exists("missing") is False
    
    # Test delete
    assert cache.delete("key1") is True
    assert cache.get("key1") is None
    assert cache.delete("missing") is False


def test_cache_data_types():
    """Test various data types."""
    
    cache = CacheLayer(use_fake=True)
    cache.clear()  # Start with clean cache
    
    # Test dict
    data = {"id": 123, "name": "test"}
    cache.set("dict", data)
    assert cache.get("dict") == data
    
    # Test list
    data = [1, 2, "three"]
    cache.set("list", data)
    assert cache.get("list") == data


if __name__ == "__main__":
    test_cache_basic_operations()
    test_cache_data_types()
    print("All tests passed!")
