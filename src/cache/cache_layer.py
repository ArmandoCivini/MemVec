"""
CacheLayer implementation using LMDB as an embedded key/value store.
"""

import pickle
from typing import Any
import lmdb


class CacheLayer:
    """
    A cache layer using LMDB for efficient key-value storage.
    Uses pickle for serialization/deserialization of Python objects.
    """
    
    def __init__(self, path: str, map_size: int = int(1e9)):
        """
        Initialize LMDB environment.
        
        Args:
            path: Directory path where LMDB database will be stored
            map_size: Maximum size of the database in bytes (default: 1GB)
        """
        self.path = path
        self.map_size = map_size
        
        # Initialize LMDB environment
        self.env = lmdb.open(
            path,
            map_size=map_size,
            writemap=True,
            map_async=True
        )
    
    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the cache using pickle serialization.
        
        Args:
            key: String key to store the value under
            value: Any Python object to store
        """
        # Serialize the value using pickle
        serialized_value = pickle.dumps(value)
        
        # Store in LMDB
        with self.env.begin(write=True) as txn:
            txn.put(key.encode('utf-8'), serialized_value)
    
    def get(self, key: str) -> Any:
        """
        Retrieve a value from the cache using pickle deserialization.
        
        Args:
            key: String key to retrieve the value for
            
        Returns:
            The deserialized Python object, or None if key not found
        """
        with self.env.begin() as txn:
            serialized_value = txn.get(key.encode('utf-8'))
            
            if serialized_value is None:
                return None
            
            # Deserialize the value using pickle
            return pickle.loads(serialized_value)
    
    def close(self) -> None:
        """
        Close the LMDB environment.
        """
        self.env.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure LMDB is properly closed."""
        self.close()
