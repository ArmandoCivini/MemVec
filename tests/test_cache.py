"""
Tests for cache integration in query path.
"""

import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.query import query_system
from src.workflow import add_file_to_system
from src.processes.process_file import FileProcessor
from src.processes.components import SentenceTransformerEmbedding
from src.index.index import HNSWIndex
from src.s3.mock_client import MockS3Client
from src.cache.cache_layer import CacheLayer
from src.vectors.pointer import Pointer


def test_cache_populates_and_is_used(tmp_path):
    """First query should populate cache; subsequent query should work even if S3 object is deleted."""
    # Setup components
    embedding_generator = SentenceTransformerEmbedding()
    processor = FileProcessor(embedding_generator)
    dimension = processor.get_index_dimension()
    index = HNSWIndex(dimension=dimension)
    s3 = MockS3Client()
    bucket = "cache-test-bucket"
    cache = CacheLayer(use_fake=True)

    # Ingest a small file
    test_file = os.path.join(os.path.dirname(__file__), "..", "datasets", "attention.pdf")
    test_file = os.path.abspath(test_file)
    assert os.path.exists(test_file), "Test PDF missing"

    add_result = add_file_to_system(
        file_path=test_file,
        processor=processor,
        index=index,
        bucket_name=bucket,
        s3_client=s3,
    )
    assert add_result["success"] is True

    # First query populates cache
    q1 = query_system(
        query_text="attention mechanism",
        index=index,
        bucket_name=bucket,
        embedding_generator=embedding_generator,
        k=3,
        s3_client=s3,
        cache=cache,
    )
    assert q1["success"] is True
    assert q1["total_found"] > 0

    # Determine chunk keys that should be cached
    # Build all chunk ids from index vector ids returned
    chunk_ids = set()
    for r in q1["search_results"]:
        chunk_ids.add(r["chunk_id"])

    # Check cache has entries
    keys = [f"chunk:{cid}" for cid in chunk_ids]
    cached = cache.batch_get(keys)
    assert set(cached.keys()) == set(keys)

    # Simulate S3 unavailability for those chunks by deleting objects
    from src.s3.chunker import create_chunk_key
    for cid in chunk_ids:
        key = create_chunk_key(cid)
        full_key = f"{bucket}/{key}"
        if full_key in s3.objects:
            del s3.objects[full_key]

    # Second query should still succeed using cache
    q2 = query_system(
        query_text="attention mechanism",
        index=index,
        bucket_name=bucket,
        embedding_generator=embedding_generator,
        k=3,
        s3_client=s3,
        cache=cache,
    )
    assert q2["success"] is True
    assert q2["total_found"] > 0

    # Distances may differ order, but vectors should exist
    for r in q2["search_results"]:
        assert isinstance(r["vector_values"], list)
        assert len(r["vector_values"]) == dimension


def test_cache_batch_ops():
    """batch_get/batch_set should round-trip values."""
    cache = CacheLayer(use_fake=True)
    items = {f"k{i}": [i, i + 1] for i in range(5)}
    cache.batch_set(items, ttl=60)
    got = cache.batch_get(list(items.keys()))
    assert got == items
