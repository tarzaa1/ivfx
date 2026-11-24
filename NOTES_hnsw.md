# HNSW + 8-bit quantization notes

Working notes for `src/hnswx.py` and `src/hnsw_playground.ipynb`.

## Goal

antirez's post on Redis Vector Sets (<https://antirez.com/news/156>) says 8-bit
quantization gives ~4x smaller vectors, an "almost 4x speedup" and "virtually
the same" recall, which is why Q8 is the Redis default. We want to check the
recall claim, and compare his per-vector Q8 against faiss SQ8 and plain
float32 HNSW. No rerank: the point is that Q8 is a low-hanging fruit on its own.

## The three indexes

| builder | storage | bytes / vector | training |
|---|---|---|---|
| `build_hnsw` | float32 | 4·d | none |
| `build_hnswsq` | faiss `QT_8bit`: one [min, max] **per dimension**, shared by all vectors, uint8 codes | d | yes (min/max from a sample) |
| `build_hnswq8` | Redis Q8: one scale **per vector**, `max|x|/127`, int8 codes in [-127, 127] | d + 4 (simulated) | none |

**faiss SQ8** keeps the query in float32 and decodes the stored codes for each
distance (asymmetric). **Redis Q8** quantizes the query too, takes the int8 dot
product, and multiplies by `scale_a * scale_b` once.

### How Q8 is simulated without touching faiss

```
<codes_a * s_a, codes_b * s_b> = s_a * s_b * <codes_a, codes_b>
```

So `build_hnswq8` quantizes every base vector, **dequantizes** it
(`codes * scale`) and adds that to a plain `IndexHNSWFlat`. The `HNSWQ8`
wrapper does the same to each query before searching. Every distance faiss
computes, during graph build and search, is then the Redis Q8 distance (checked
to ~1e-7 relative error). `quantize_queries=False` gives float queries against
Q8 vectors, to separate the two error sources.

**What this does not give:** the memory and speed benefit. faiss still stores
and computes float32, so `hnswq8` costs the same as `hnsw`. `code_size` and
`index_size()["vectors"]` report what real Q8 storage would be (d + 4 B).
Getting real int8 storage needs a custom faiss `DistanceComputer` (C++), a
library with native int8 HNSW, or benchmarking Redis itself (`VADD` default
vs `NOQUANT`).

Differences from Redis kept on purpose: Redis always normalizes and uses
cosine; here data is used as loaded, with `"l2"` or `"ip"`. Rounding is
round-to-nearest (the post does not say). The post's `range = 2*max_abs`
comment only changes a constant factor, not the ranking.

## Findings

### Memory: the graph does not shrink

With M=32, the graph is 2·M = 64 int32 neighbor ids per node on level 0, plus
upper levels, the same for every storage type. Quantization only shrinks the
vectors, so the whole index shrinks far less than 4x at low d (antirez says
the same).

GloVe (1.18M × 100, M=32), from `index_size`:

| index | vectors | graph | total | serialized |
|---|---|---|---|---|
| hnsw | 451.5 MiB | 307.2 MiB | 758.6 MiB | 758.6 MiB |
| hnswsq | 112.9 MiB | 307.2 MiB | 420.0 MiB | 420.0 MiB |
| hnswq8 | 117.4 MiB | 307.2 MiB | 424.6 MiB | 758.6 MiB (float32 simulation) |

`faiss.serialize_index(index).nbytes` matches vectors + graph exactly for the
faiss indexes.

### Speed: faiss SQ8 needs `d % 8 == 0`

On GloVe (d=100) faiss SQ8 built 3.5x slower and searched ~2x slower than
float32. Cause: faiss's SIMD SQ path only runs when d is a multiple of 8
(AVX2 build); otherwise every component is decoded one at a time. The metric
is irrelevant.

Brute force, 300k GloVe vectors, 10 queries:

| d | metric | flat | SQ8 |
|---|---|---|---|
| 100 | l2 | 28.9 ms | 81.0 ms |
| 100 | ip | 26.5 ms | 66.2 ms |
| 104 (zero-padded) | l2 | 24.7 ms | 24.4 ms |
| 104 (zero-padded) | ip | 23.1 ms | 10.2 ms |

HNSW on 200k GloVe, ip: d=100 SQ8 build 18.9 s vs flat 7.4 s; padded to 104,
SQ8 build 5.1 s vs flat 7.5 s, with unchanged recall (0.8394 vs 0.8440 flat).
Zero padding changes no distance. **Fix: pad to `(d + 7) // 8 * 8`** (not
implemented yet, see next steps).

With d a multiple of 8, SQ8 build time is about equal to flat at scale
(1M SIFT: 44.8 s vs 42.8 s; arxiv: 179.2 s vs 180.7 s). At 100k SIFT SQ8 was
faster (1.6 s vs 2.8 s), most likely because 12 MiB of codes fit in L3 cache
and 49 MiB of floats do not. Training is negligible (0.03 s on 300k).

Single search timings (one run of 1000 queries) are noisy, ±2x between
identical indexes. Repeat searches and take the minimum before comparing.

### Recall (M=32, efConstruction=40, efSearch=64, k=10, no rerank)

| dataset | hnsw | hnswsq | hnswq8 |
|---|---|---|---|
| SIFT 100k (l2) | 0.9944 | 0.9832 | 0.9809 |
| GloVe 1.18M (ip) | 0.7944 | 0.7837 | 0.7801 |
| arxiv (d=768, ip) | 0.9859 | 0.9751 | not run yet |

So far both 8-bit schemes cost ~1 point of recall@10, Q8 slightly more than
SQ8.

### Dataset caveats

- **SIFT** components are integers 0..218. `QT_8bit_direct` stores SIFT in 1 B
  per dimension **losslessly** with no training. Q8's symmetric range also
  wastes its sign bit on SIFT (only 0..127 is used, effectively 7 bits). SIFT
  is not a fair test of either quantizer.
- The faithful Q8 test is on unit-norm, signed data (glove, coco, imagenet,
  arxiv) with `metric="ip"`, which is what Redis does (cosine).

### faiss 8-bit types (1.16)

| qtype | needs train | encoding |
|---|---|---|
| `QT_8bit` | yes | per-dimension min/max |
| `QT_8bit_uniform` | yes | one global min/max |
| `QT_8bit_direct` | no | value as-is, integers 0..255 |
| `QT_8bit_direct_signed` | no | value as-is, integers -128..127 |

None does per-vector scaling, so none reproduces Q8. Q8's real advantage over
faiss SQ8 is not needing a training sample (no range drift for streaming
inserts).

## Next steps

- Add zero-padding to `build_hnswsq` (`RemapDimensionsTransform` in an
  `IndexPreTransform`, then unwrap in `search` / `code_size` / `index_size`).
- Run Q8 on arxiv, coco, imagenet; try `quantize_queries=False`.
- Repeat search timings and take the minimum.
- Real Q8 speed/memory: benchmark Redis Vector Sets (`VADD` Q8 vs `NOQUANT`).
