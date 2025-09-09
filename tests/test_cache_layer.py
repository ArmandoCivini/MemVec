"""
Simple tests for CacheLayer functionality.
"""

import sys
import os
import tempfile
import numpy as np

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache.cache_layer import CacheLayer


def test_cache_basic_operations():
    """Test basic set/get operations."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = os.path.join(temp_dir, "test_cache")
        
        with CacheLayer(cache_path) as cache:
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


def test_cache_data_types():
    """Test various data types."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = os.path.join(temp_dir, "test_cache")
        
        with CacheLayer(cache_path) as cache:
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
