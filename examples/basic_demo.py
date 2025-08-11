"""
Example script demonstrating basic MemVec usage.

This script shows how to:
1. Initialize the MemVec system
2. Add vectors to the system
3. Perform searches
4. Monitor cache performance
"""

import numpy as np
import time
from src.main import MemVecRouter


def generate_random_vectors(count: int, dimension: int) -> dict:
    """Generate random test vectors."""
    vectors = {}
    for i in range(count):
        vector_id = f"vec_{i:06d}"
        vector = np.random.random(dimension).astype(np.float32)
        vectors[vector_id] = vector
    return vectors


def main():
    """Run the basic MemVec demo."""
    print("MemVec Basic Demo")
    print("=================")
    
    # Configuration
    DIMENSION = 128
    VECTOR_COUNT = 1000
    QUERY_COUNT = 10
    
    print(f"Initializing MemVec with {DIMENSION}-dimensional vectors...")
    
    # TODO: Uncomment when implementation is ready
    # router = MemVecRouter(dimension=DIMENSION)
    
    print(f"Generating {VECTOR_COUNT} random test vectors...")
    test_vectors = generate_random_vectors(VECTOR_COUNT, DIMENSION)
    
    print("Adding vectors to the system...")
    # TODO: Uncomment when implementation is ready
    # router.add_vectors(test_vectors)
    
    print(f"Running {QUERY_COUNT} search queries...")
    for i in range(QUERY_COUNT):
        query_vector = np.random.random(DIMENSION).astype(np.float32)
        
        start_time = time.time()
        # TODO: Uncomment when implementation is ready
        # results = router.search(query_vector, k=10)
        end_time = time.time()
        
        # TODO: Uncomment when implementation is ready
        # print(f"Query {i+1}: Found {len(results)} results in {(end_time - start_time)*1000:.2f}ms")
        print(f"Query {i+1}: [Implementation pending]")
    
    print("\nPerformance Metrics:")
    # TODO: Uncomment when implementation is ready
    # metrics = router.get_metrics()
    # for metric, value in metrics.items():
    #     print(f"  {metric}: {value}")
    print("  [Metrics implementation pending]")


if __name__ == "__main__":
    main()
