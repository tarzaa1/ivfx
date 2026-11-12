import numpy as np
import faiss

METRICS = {
    "l2": faiss.METRIC_L2,
    "ip": faiss.METRIC_INNER_PRODUCT,
}


def build_ivf(d, base, metric, nlist):
    """
    Build an IndexIVFFlat with a flat quantizer, train it on `base`, add `base`.

    d      : vector dimension
    base   : (N, d) float32
    metric : "l2" or "ip"
    nlist  : number of inverted lists (coarse centroids)
    """
    metric_type = METRICS[metric]
    if metric_type == faiss.METRIC_L2:
        quantizer = faiss.IndexFlatL2(d)
    else:
        quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, metric_type)
    index.train(base)
    index.add(base)
    return index


def build_ivfpq(d, base, metric, nlist, m, nbits=8):
    """
    Build an IndexIVFPQ with a flat quantizer, train it on `base`, add `base`.

    d      : vector dimension (must be divisible by m)
    base   : (N, d) float32
    metric : "l2" or "ip"
    nlist  : number of inverted lists (coarse centroids)
    m      : number of PQ sub-quantizers (code size in bytes when nbits=8)
    nbits  : bits per sub-quantizer (256 centroids each at the default 8)
    """
    metric_type = METRICS[metric]
    if metric_type == faiss.METRIC_L2:
        quantizer = faiss.IndexFlatL2(d)
    else:
        quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, metric_type)
    index.train(base)
    index.add(base)
    return index


def build_ivfrabitq(d, base, metric, nlist, nb_bits=1, qb=None):
    """
    Build an IndexIVFRaBitQ with a flat quantizer, train it on `base`, add `base`.

    d       : vector dimension
    base    : (N, d) float32
    metric  : "l2" or "ip"
    nlist   : number of inverted lists (coarse centroids)
    nb_bits : bits per dimension for the stored codes (1 = classic RaBitQ,
              code size ~ d/8 bytes plus a few bytes of per-vector factors)
    qb      : bits used to quantize the query at search time (faiss default 4).
              Higher is more accurate, slower; 0 falls back to the unquantized query.
    """
    metric_type = METRICS[metric]
    if metric_type == faiss.METRIC_L2:
        quantizer = faiss.IndexFlatL2(d)
    else:
        quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFRaBitQ(quantizer, d, nlist, metric_type, True, nb_bits)
    if qb is not None:
        index.qb = qb
    index.train(base)
    index.add(base)
    return index


# (bits per component, use QJL bit) -> ScalarQuantizer type
_SQ = faiss.ScalarQuantizer
TURBOQUANT_QTYPES = {
    (1, False): _SQ.QT_1bit_tqmse,
    (2, False): _SQ.QT_2bit_tqmse,
    (3, False): _SQ.QT_3bit_tqmse,
    (4, False): _SQ.QT_4bit_tqmse,
    (8, False): _SQ.QT_8bit_tqmse,
    (2, True): _SQ.QT_2bit_tq,   # 1-bit MSE + 1-bit QJL + factors
    (3, True): _SQ.QT_3bit_tq,   # 2-bit MSE + 1-bit QJL + factors
    (4, True): _SQ.QT_4bit_tq,   # 3-bit MSE + 1-bit QJL + factors
    (5, True): _SQ.QT_5bit_tq,   # 4-bit MSE + 1-bit QJL + factors
}


def build_ivfturboquant(d, base, metric, nlist, nbits=4, qjl=False, scale=None):
    """
    Build an IVF index whose lists are encoded with TurboQuant (a data-oblivious
    scalar quantizer), train it on `base`, add `base`.

    d      : vector dimension
    base   : (N, d) float32
    metric : "l2" or "ip".  NOTE: in faiss 1.16 the inner-product scan for the
             TurboQuant types ranks much worse than L2 on identical data; on
             unit-norm vectors L2 gives the same ranking, so prefer "l2".
    nlist  : number of inverted lists (coarse centroids)
    nbits  : bits per component. MSE-only (qjl=False): 1, 2, 3, 4, 8.
             With qjl=True: 2, 3, 4, 5 (one of the bits is a QJL sign bit).
             Code size is d * nbits / 8 bytes (+ a few factor bytes with qjl).
    qjl    : use the "full" TurboQuant types with a QJL bit (QT_*bit_tq).
             These scored below the MSE-only types at the same bits in our tests.
    scale  : TurboQuant's codebook is fixed for ~unit-norm inputs and has no
             range training, so raw-scale data (e.g. sift, norms ~500) must be
             scaled.  None = no scaling (fine for unit-norm data); "auto" =
             1 / mean vector norm; or a float.  A global scale keeps the L2
             ranking, so results stay comparable.  When scale is set, the
             returned index is an IndexPreTransform wrapping the IVF index;
             `search` handles this.
    """
    metric_type = METRICS[metric]
    qtype = TURBOQUANT_QTYPES[(nbits, qjl)]
    if metric_type == faiss.METRIC_L2:
        quantizer = faiss.IndexFlatL2(d)
    else:
        quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFScalarQuantizer(quantizer, d, nlist, qtype, metric_type)

    if scale is not None:
        if scale == "auto":
            scale = 1.0 / float(np.linalg.norm(base[:100_000], axis=1).mean())
        lt = faiss.LinearTransform(d, d, False)
        faiss.copy_array_to_vector((np.eye(d, dtype=np.float32) * scale).ravel(), lt.A)
        lt.is_trained = True
        index = faiss.IndexPreTransform(lt, index)

    index.train(base)
    index.add(base)
    return index

def search(index, queries, k, nprobe=None):
    """
    Search `index` for the top-k neighbors of each query.
    Optionally set nprobe (number of lists scanned per query).
    Returns (D, I): distances (nq, k) and ids (nq, k).
    """
    if nprobe is not None:
        # works for a bare IVF index and for one wrapped in IndexPreTransform
        faiss.extract_index_ivf(index).nprobe = nprobe
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


def rerank(base, queries, I, k, metric):
    """
    Exact rerank of candidate ids.

    base    : (N, d) float32 — full base vectors
    queries : (nq, d) float32
    I       : (nq, c) candidate ids per query (c >= k), e.g. from an IVF-PQ search
    k       : number of results to keep per query
    metric  : "l2" or "ip"

    For each query, builds a flat index over its candidate vectors, searches it
    exactly, and maps the local results back to base ids.
    Returns (D, I_new): distances (nq, k) and ids (nq, k).
    """
    nq, c = I.shape
    if c < k:
        raise ValueError(f"need at least k={k} candidates per query, got {c}")
    d = base.shape[1]
    flat_cls = faiss.IndexFlatL2 if METRICS[metric] == faiss.METRIC_L2 else faiss.IndexFlatIP

    D_new = np.empty((nq, k), dtype=np.float32)
    I_new = np.full((nq, k), -1, dtype=np.int64)
    for q in range(nq):
        cand = I[q][I[q] >= 0]  # drop faiss padding (-1) if a list ran short
        flat = flat_cls(d)
        flat.add(base[cand])
        Dq, Iq = flat.search(queries[q : q + 1], min(k, len(cand)))
        n = Iq.shape[1]
        D_new[q, :n] = Dq[0]
        I_new[q, :n] = cand[Iq[0]]
    return D_new, I_new
