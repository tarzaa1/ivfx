import numpy as np
import faiss

METRICS = {
    "l2": faiss.METRIC_L2,
    "ip": faiss.METRIC_INNER_PRODUCT,
}


def build_hnsw(d, base, metric, M=32, ef_construction=40):
    """
    Build an IndexHNSWFlat (float32 vectors, no quantization) and add `base`.

    d               : vector dimension
    base            : (N, d) float32
    metric          : "l2" or "ip"
    M               : graph degree (neighbors per node; 2*M on level 0)
    ef_construction : beam width while inserting
    """
    index = faiss.IndexHNSWFlat(d, M, METRICS[metric])
    index.hnsw.efConstruction = ef_construction
    index.add(base)
    return index


def build_hnswsq(d, base, metric, M=32, ef_construction=40, qtype=faiss.ScalarQuantizer.QT_8bit):
    """
    Build an IndexHNSWSQ (faiss scalar quantizer storage), train it on `base`, add `base`.

    faiss SQ8 learns one [min, max] range per dimension from the training data
    (shared by all vectors) and stores each component as a uint8 in that range:
    d bytes per vector. Queries stay float32; distances are computed between the
    float query and the decoded vectors (asymmetric).
    """
    index = faiss.IndexHNSWSQ(d, qtype, M, METRICS[metric])
    index.hnsw.efConstruction = ef_construction
    index.train(base)
    index.add(base)
    return index


def quantize_q8(x):
    """
    Redis vector-set style Q8: per-vector symmetric int8 quantization.

    x : (N, d) float32
    Returns (codes, scales): codes (N, d) int8 in [-127, 127] and scales (N,)
    float32 with x ~= codes * scales[:, None]. The scale is max|x_i| / 127 for
    each vector, so every vector uses the full int8 range on its own.
    """
    x = np.asarray(x, dtype=np.float32)
    max_abs = np.abs(x).max(axis=1)
    scales = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
    codes = np.clip(np.rint(x / scales[:, None]), -127, 127).astype(np.int8)
    return codes, scales


def dequantize_q8(codes, scales):
    """Inverse of quantize_q8: (N, d) float32 = codes * scales."""
    return codes.astype(np.float32) * scales[:, None]


class HNSWQ8:
    """
    Thin wrapper around an IndexHNSWFlat holding dequantized Q8 vectors.
    Quantizes the queries before searching (when quantize_queries is set).
    `hnsw` is the underlying faiss HNSW structure, so efSearch works as usual.
    """

    def __init__(self, index, quantize_queries):
        self.index = index
        self.hnsw = index.hnsw
        self.quantize_queries = quantize_queries
        self.code_size = index.d + 4  # real Q8 bytes per vector: int8 codes + float scale

    @property
    def ntotal(self):
        return self.index.ntotal

    def search(self, queries, k):
        if self.quantize_queries:
            queries = dequantize_q8(*quantize_q8(queries))
        return self.index.search(queries, k)


def build_hnswq8(d, base, metric, M=32, ef_construction=40, quantize_queries=True):
    """
    Build an HNSW index that behaves like Redis's Q8 vectors, without touching
    the faiss HNSW code. Returns an HNSWQ8 wrapper.

    Each base vector is quantized with quantize_q8 and the *dequantized* vector
    (codes * scale) is added to a plain IndexHNSWFlat. Because
        <codes_a * s_a, codes_b * s_b> = s_a * s_b * <codes_a, codes_b>,
    every distance faiss computes (during graph construction and search) is
    exactly the rescaled int8 dot product Redis computes, up to float32
    rounding. With quantize_queries=True the queries are quantized the same
    way at search time, as Redis does; False keeps them float32 (asymmetric).

    This only simulates the accuracy: faiss still stores float32. The real
    storage cost is d + 4 bytes per vector (int8 codes + one float scale),
    which is what `code_size(index)` reports for it.

    Redis normalizes vectors and uses cosine. Here the data is used as given:
    on unit-norm data use "ip" or "l2"; on sift use "l2".
    """
    codes, scales = quantize_q8(base)
    index = faiss.IndexHNSWFlat(d, M, METRICS[metric])
    index.hnsw.efConstruction = ef_construction
    index.add(dequantize_q8(codes, scales))
    return HNSWQ8(index, quantize_queries)


def code_size(index):
    """Bytes stored per vector (excluding the graph links)."""
    if isinstance(index, HNSWQ8):
        return index.code_size
    return faiss.downcast_index(index.storage).code_size


def index_size(index):
    """
    Size in bytes of a built HNSW index, as a dict:
      vectors    : stored vectors / codes (ntotal * bytes per vector)
      graph      : HNSW link structure (neighbor ids, per-node offsets, levels)
      total      : vectors + graph
      serialized : exact size of faiss.serialize_index (everything faiss writes
                   to disk; ~ the RAM it holds, minus allocator slack)

    For HNSWQ8 the faiss index holds float32 vectors, so `serialized` is what
    this simulation really uses, while `vectors` / `total` are what Redis-style
    Q8 storage (d + 4 bytes per vector) would take with the same graph.
    """
    faiss_index = index.index if isinstance(index, HNSWQ8) else index
    hnsw = faiss_index.hnsw
    graph = (
        hnsw.neighbors.size() * 4    # int32 neighbor ids, all levels
        + hnsw.offsets.size() * 8    # size_t start of each node's neighbor list
        + hnsw.levels.size() * 4     # int32 level of each node
    )
    vectors = faiss_index.ntotal * code_size(index)
    return {
        "vectors": vectors,
        "graph": graph,
        "total": vectors + graph,
        "serialized": faiss.serialize_index(faiss_index).nbytes,
    }


def search(index, queries, k, ef_search=None):
    """
    Search `index` for the top-k neighbors of each query.
    Optionally set ef_search (beam width at query time, should be >= k).
    Works for faiss HNSW indexes and for HNSWQ8 (which quantizes the queries).
    Returns (D, I): distances (nq, k) and ids (nq, k).
    """
    if ef_search is not None:
        index.hnsw.efSearch = ef_search
    return index.search(queries, k)


def recall(I, gt, k):
    """
    Recall@k: fraction of the true top-k (gt[:, :k]) found in I[:, :k], averaged over queries.
    """
    nq = gt.shape[0]
    hits = 0
    for q in range(nq):
        hits += len(set(I[q, :k].tolist()) & set(gt[q, :k].tolist()))
    return hits / (nq * k)
