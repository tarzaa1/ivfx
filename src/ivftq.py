import numpy as np
import faiss

METRICS = {
    "l2": faiss.METRIC_L2,
    "ip": faiss.METRIC_INNER_PRODUCT,
}

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