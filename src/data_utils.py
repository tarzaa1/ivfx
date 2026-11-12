"""
data_utils.py — Load datasets from ../edge/data into numpy arrays.
"""

import os
import numpy as np
import faiss

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "edge", "data")

NQ = 1000  # number of queries returned per dataset
K_GT = 100  # ground-truth neighbors per query

# name -> (HDF5 filename, L2-normalize base and queries on load)
HDF5_DATASETS = {
    "glove": ("glove-100-angular.hdf5", True),
    "coco": ("coco-i2i-512-angular.hdf5", True),
    "imagenet": ("imagenet-clip-512-normalized.hdf5", False),
    "arxiv": ("arxiv-nomic-768-normalized.hdf5", False),
}


def read_fvecs(path):
    """Read a .fvecs file -> (N, d) float32 array."""
    data = np.fromfile(path, dtype=np.int32)
    d = data[0]
    return data.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()


def read_ivecs(path):
    """Read a .ivecs file -> (N, d) int32 array."""
    data = np.fromfile(path, dtype=np.int32)
    d = data[0]
    return data.reshape(-1, d + 1)[:, 1:].copy()


def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def compute_gt_l2(base, queries, k=K_GT):
    """Brute-force L2 ground truth -> (nq, k) int64."""
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base)
    _, gt = index.search(queries, k)
    return gt.astype(np.int64)


def load_hdf5(path, nq=NQ):
    """Load an ann-benchmarks HDF5 file -> (base, first nq queries, their gt)."""
    import h5py
    with h5py.File(path, "r") as f:
        base = np.asarray(f["train"], dtype=np.float32)
        queries = np.asarray(f["test"][:nq], dtype=np.float32)
        gt = np.asarray(f["neighbors"][:nq], dtype=np.int64)
    return base, queries, gt


def load_dataset(name, data_root=DATA_ROOT, nq=NQ):
    """
    Load a dataset by name into numpy arrays.

    Returns (base, queries, gt):
      base    (N, d)    float32
      queries (nq, d)   float32  — the first nq queries
      gt      (nq, 100) int64    — ground-truth neighbor ids for those queries

    glove / coco are L2-normalized on load (their ground truth is angular).
    imagenet / arxiv are already unit-norm and are returned as-is.
    sift is raw (L2 metric); its ground truth is computed here with IndexFlatL2.
    """
    if name in HDF5_DATASETS:
        fname, normalize = HDF5_DATASETS[name]
        base, queries, gt = load_hdf5(os.path.join(data_root, fname), nq)
        if normalize:
            base = l2_normalize(base)
            queries = l2_normalize(queries)
        return base, queries, gt

    base = read_fvecs(os.path.join(data_root, f"{name}_base.fvecs"))
    queries = read_fvecs(os.path.join(data_root, f"{name}_query.fvecs"))[:nq]
    gt = compute_gt_l2(base, queries)
    return base, queries, gt
