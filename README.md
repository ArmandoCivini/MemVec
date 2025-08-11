# VecCache — S3-backed Hot Vector Cache

VecCache is a small prototype project that implements a fast in-memory cache layer for large vector collections stored cheaply in Amazon S3 Vectors. The goal: give real-time similarity search latency for the common "hot" queries while keeping the full knowledge base in low-cost S3 storage.

---

# Elevator pitch (for non-technical / high-level readers)

Imagine your knowledge base is huge — billions of embeddings — and most of it sits in cheap storage. But your application (a chatbot, recommender, or semantic search) needs instant answers for the common queries. VecCache keeps a small, fast copy of the most-used vectors in memory so queries return instantly. When the in-memory copy doesn’t have the data, VecCache fetches the missing pieces from S3 and fills the cache. The result: fast responses where it matters, affordable storage where it doesn’t.

---

# Project goals

* Demonstrate a practical two-tier vector architecture: **hot in-memory cache** + **cold S3 Vectors**.
* Provide a minimal, hackable prototype you can extend: uses an ANN index to retrieve candidate IDs, checks a memory cache for raw vectors, and falls back to S3 Vectors on misses.
* Measure the trade-offs: latency, hit-rate, memory usage, and S3 access costs.

---

# Key ideas (short)

* Keep a small percentage of vectors (the hot set) in RAM for sub-100ms responses.
* Use an ANN index (FAISS / JECQ / HNSW) to select candidate IDs for a query.
* For each candidate ID, check the cache (Redis/Valkey or local memory); on misses, fetch vectors from S3 Vectors and populate the cache.
* Evict cached vectors using LRU/LFU or a hybrid policy; optionally pin very-hot items.

---

# High-level architecture (summary)

1. Client sends a query (text or embedding) to the Query Router.
2. Query Router transforms text → embedding (if needed), then runs ANN search on an index of IDs (FAISS or similar).
3. The ANN index returns candidate IDs (e.g., 200 IDs).
4. Router checks the in-memory cache for the raw vectors of those IDs.

   * If the cache contains the vectors, respond quickly.
   * If some or all are missing, call S3 Vectors (QueryVectors / GetVectors) to fetch raw vectors, then re-rank and return results.
5. Fetched vectors get inserted into the cache according to the eviction policy.

**TODO:** add high-level architecture diagram (Query Router → ANN index → Cache → S3 Vectors).

---

# Why this makes sense

* Traffic is usually Zipfian: a small fraction of items produces most queries, so a small cache yields large latency improvements.
* S3 Vectors is cost-effective for cold storage; RAM is expensive but very fast. Two-tiering optimizes cost vs latency.
* The index and the raw vectors are separated: you can keep a compact index in memory and store raw vectors in S3 until needed.

---

# Components & tech choices (prototype suggestions)

* **Query Router / Orchestrator**: Python (FastAPI) or Go for low-latency RPC.
* **ANN Index**: FAISS (Python bindings) or JECQ (if available) for fast candidate selection.
* **Hot Cache**: Redis with vector module (or Valkey-Search) or local in-process memory for simplest POC.
* **Cold Store**: Amazon S3 Vectors (or plain S3 / MinIO for offline testing).
* **Embeddings**: any embedding model (open models or cloud provider); for POC you can use random or public datasets.

Trade-offs:

* Python is fine for prototyping: FAISS is C++ under the hood and `redis-py` is simple.
* If you need extreme throughput later, move parts to Go or Rust or use a compiled ANN library directly.

---

# Core algorithms to implement

1. **ANN search step**

   * Use FAISS/HNSW to get candidate IDs for each query.

2. **Cache lookup & miss handling**

   * For candidate IDs returned by ANN, perform a multi-get in Redis for those IDs.
   * For missing IDs, call S3 Vectors `GetVectors` or `QueryVectors?returnData=true`.
   * Re-rank using exact similarity on the retrieved vectors.

3. **Eviction policy**

   * Start with Redis-managed LRU/LFU or TTL-based eviction.
   * Optional advanced policies: frequency+recency hybrid, pinning, or popularity-forecasting.

4. **Quality thresholding**

   * If the top cached candidates have similarity > X (a tunable threshold), you may short-circuit and skip fetching from S3 to save cost and latency.
   * Otherwise, fetch additional vectors from S3 and re-rank.

---

# Minimal data flow (quick)

1. Query → embedding
2. ANN(index) → candidate IDs
3. Cache (multi-get) → some vectors
4. If misses → S3 Vectors fetch
5. Re-rank and return
6. Insert fetched vectors into cache (with TTL)

**TODO:** add sequence diagram showing the exact RPC calls and their expected latencies.

---

# Testing & benchmark plan (POC-level)

1. **Dataset**: generate 100k–1M vectors (128–512 dims) or use public datasets (SIFT, GloVe, LAION embeddings, etc.).
2. **Simulate load**: produce query traffic with a Zipfian distribution so hot items appear frequently.
3. **Metrics to capture**:

   * Query latency (P50, P95, P99)
   * Cache hit rate
   * S3 read count and bandwidth
   * Memory usage
4. **Experiments**:

   * Vary cache size (1%, 5%, 20%) and measure hit rate and latency.
   * Vary ANN candidate count (e.g., 50, 200, 500) to see the trade-off between recall and S3 fetches.
   * Measure behavior with and without short-circuit thresholding.

**TODO:** add example benchmark scripts & result plots.

---

# Quickstart (local POC)

1. Install dependencies (Python):

```bash
python -m venv .venv
source .venv/bin/activate
pip install faiss-cpu redis boto3 fastapi uvicorn
```

2. Run a local Redis (Docker):

```bash
docker run -p 6379:6379 redis:7
```

3. Start a basic FAISS indexer and minimal router (scripts in `/examples`).

4. Populate S3 (or MinIO) with vectors for the dataset and run the benchmark harness `bench/run_load_test.py`.

**TODO:** fill in example scripts & CLI commands in `/examples`.

---

# Roadmap / TODOs

*

**TODO:** Add architecture diagrams (high-level and deployment), sequence diagrams, benchmark plots, and example code links.

---

# Contributions & License

This project is a weekend POC. Contributions welcome — open issues for ideas or improvements. Consider MIT license.

---

# Acknowledgements & References

* S3 Vectors docs and AWS blog posts (for design patterns and API features)
* FAISS, HNSW, and ANN literature for indexing choices
* Redis / Valkey for in-memory vector search modules

**TODO:** add formal references and link to the specific AWS docs used during research.

---

*Created as a concise starter README for a POC vector cache leveraging S3 Vectors and an in-memory ANN/cache layer.*
