# ivfq

Small FAISS playground for comparing inverted-file (IVF) indexes with different
fine encodings: flat, product quantization (PQ) and RaBitQ.

## Layout

```
src/data_utils.py   load datasets from ../edge/data into numpy arrays
src/ivfx.py         build / search / rerank / recall helpers
src/playground.ipynb
requirements.txt
```

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Datasets are read from `../edge/data` (ann-benchmarks HDF5 files plus SIFT
`.fvecs`). `load_dataset(name)` returns `(base, queries, gt)`:

| name       | d   | notes                                              |
|------------|-----|----------------------------------------------------|
| `sift`     | 128 | raw L2 data; ground truth computed with IndexFlatL2 |
| `glove`    | 100 | angular; L2-normalized on load                     |
| `coco`     | 512 | angular; L2-normalized on load                     |
| `imagenet` | 512 | already unit norm                                  |
| `arxiv`    | 768 | already unit norm                                  |

Queries are truncated to the first 1000, ground truth is top-100 ids.

## Helpers (`src/ivfx.py`)

```python
build_ivf(d, base, metric, nlist)                    # IndexIVFFlat
build_ivfpq(d, base, metric, nlist, m, nbits=8)      # IndexIVFPQ
build_ivfrabitq(d, base, metric, nlist, nb_bits=1)   # IndexIVFRaBitQ
search(index, queries, k, nprobe=None)  -> (D, I)
rerank(base, queries, I, k, metric)     -> (D, I)    # exact rerank of candidates
recall(I, gt, k)                        -> float
```

`metric` is `"l2"` or `"ip"`. On unit-norm data both give the same ranking;
the choice then only affects the coarse k-means partition. SIFT is not unit
norm and must be built with `"l2"`.

## How the two encoders work

**IVF-PQ.** Each vector is split into `m` contiguous sub-vectors of `d/m`
dimensions. Every sub-vector is replaced by the id of its nearest centroid out
of `2^nbits` centroids learned by k-means on that sub-space. With `nbits=8`
the code is `m` bytes. Distances are approximated by summing per-sub-space
lookup tables that are computed once per query. The codebooks are learned from
the data, which is why PQ does well on structured data at low dimension.

**IVF-RaBitQ.** The residual `x - c` to the list centroid is randomly rotated
and reduced to one sign bit per dimension (`nb_bits=1`), giving `d/8` bytes.
Two extra float32 values are stored per vector: the squared residual norm
`||x - c||^2` and a rescaling factor that turns the bitwise popcount against
the quantized query into an unbiased estimate of the inner product. These are
the "+8 bytes". The code is data-oblivious (no learned codebook) and its error
shrinks with `1/sqrt(d)`, so it does best at high dimension.

## Making the comparison fair

Compare at **equal bits spent on the code**, and report RaBitQ's 8 factor
bytes as a fixed overhead:

```
PQ bytes      = m * nbits / 8        (= m at nbits=8)
RaBitQ bytes  = d * nb_bits / 8 + 8

equal code bits  <=>  m = d * nb_bits / 8
```

`m` must divide `d`. Examples:

| d   | RaBitQ 1-bit | PQ match | RaBitQ 2-bit | PQ match | RaBitQ 4-bit | PQ match |
|-----|--------------|----------|--------------|----------|--------------|----------|
| 96  | 12 + 8 B     | m=12     | 24 + 8 B     | m=24     | 48 + 8 B     | m=48     |
| 100 | 13 + 8 B     | (12.5)   | 25 + 8 B     | m=25     | 50 + 8 B     | m=50     |
| 128 | 16 + 8 B     | m=16     | 32 + 8 B     | m=32     | 64 + 8 B     | m=64     |
| 512 | 64 + 8 B     | m=64     | 128 + 8 B    | m=128    | 256 + 8 B    | m=256    |

When no `m` matches exactly (d=100, 1-bit), bracket RaBitQ with the PQ sizes
just below and above (m=10 and m=20). The 8-byte overhead matters most at
low `d`: it is 40% of the total at d=96 and 11% at d=512.

Everything else must be held constant across the two indexes:

- same `nlist` and `nprobe` (FAISS seeds k-means deterministically, so the two
  indexes get identical centroids and probe identical lists);
- same metric and the same normalization of the data;
- same number of candidates `c` fed to `rerank`, since both encoders are
  mostly used as candidate generators;
- report both pre-rerank recall@k (encoder quality) and post-rerank recall@k
  (end-to-end), plus `index.code_size` for the bytes actually stored.

Rule of thumb from our runs: at d=128 (SIFT) PQ wins at equal bits, at d=512
(COCO, ImageNet) RaBitQ wins by a few points before rerank; after reranking
100 candidates both land within a point of IVF-Flat.
