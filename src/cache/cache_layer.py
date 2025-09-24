"""
CacheLayer implementation using Redis/FakeRedis as an embedded key/value store.
"""

import pickle
from typing import Any, Optional, List, Dict
import redis
import fakeredis


class CacheLayer:
    """
    A cache layer using Redis for efficient key-value storage.
    Uses pickle for serialization/deserialization of Python objects.
    Can use either real Redis or FakeRedis for testing/development.
    """
    
    def __init__(self, use_fake: bool = True, host: str = "localhost", port: int = 6379, db: int = 0):
        """
        Initialize Redis client.
        
        Args:
            use_fake: If True, use FakeRedis for testing/development
            host: Redis server host (ignored if use_fake=True)
            port: Redis server port (ignored if use_fake=True)
            db: Redis database number
        """
        if use_fake:
            self.client = fakeredis.FakeRedis(db=db, decode_responses=False)
        else:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store a value in the cache using pickle serialization.
        
        Args:
            key: String key to store the value under
            value: Any Python object to store
            ttl: Time to live in seconds (optional)
        """
        serialized_value = pickle.dumps(value)
        self.client.set(key, serialized_value, ex=ttl)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache using pickle deserialization.
        
        Args:
            key: String key to retrieve the value for
            
        Returns:
            The deserialized Python object, or None if key not found
        """
        serialized_value = self.client.get(key)
        
        if serialized_value is None:
            return None
        
        return pickle.loads(serialized_value)
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: String key to delete
            
        Returns:
            True if key was deleted, False if key didn't exist
        """
        return bool(self.client.delete(key))
    
    def clear(self) -> None:
        """
        Clear all keys from the cache.
        """
        self.client.flushdb()
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: String key to check
            
        Returns:
            True if key exists, False otherwise
        """
        return bool(self.client.exists(key))

    # -------- Batch operations (using Redis pipeline) --------
    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """
        Retrieve multiple keys using a pipeline.

        Args:
            keys: List of keys to retrieve

        Returns:
            Dict mapping key -> deserialized value for keys that exist
        """
        if not keys:
            return {}

        pipe = self.client.pipeline()
        for key in keys:
            pipe.get(key)
        results = pipe.execute()

        out: Dict[str, Any] = {}
        for key, serialized_value in zip(keys, results):
            if serialized_value is not None:
                out[key] = pickle.loads(serialized_value)
        return out

    def batch_set(self, items: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """
        Set multiple key/value pairs using a pipeline.

        Args:
            items: Mapping of key -> Python object
            ttl: Optional TTL in seconds applied per key
        """
        if not items:
            return

        pipe = self.client.pipeline()
        for key, value in items.items():
            pipe.set(key, pickle.dumps(value), ex=ttl)
        pipe.execute()
