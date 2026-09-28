"""Input variants.

The extraction stores one row per (variant, position), so any later experiment
about *how the model responds to inputs* can be run on the cache instead of a
new forward pass.  Variants are generated deterministically from the window's
own sequence and a seed, so a run is reproducible from the manifest alone.

Built-in variants
-----------------
real            the reference sequence
random          uniform A/C/G/T, same length
polyA           constant A
shuffle_mono    mononucleotide-preserving permutation
shuffle_di      dinucleotide-preserving permutation (Altschul-Erikson)
revcomp         reverse complement of the real window
mask_center     real, with the central k bp replaced by N-free random sequence

``real``, ``random``, ``polyA`` and ``shuffle_di`` reproduce the input controls
used in the handoff manuscript; the rest are there so that later perturbation
work does not need a new extraction pass.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

BASES = np.array([ord("A"), ord("C"), ord("G"), ord("T")], dtype=np.uint8)
COMPLEMENT = {ord("A"): ord("T"), ord("C"): ord("G"), ord("G"): ord("C"),
              ord("T"): ord("A"), ord("N"): ord("N")}


def _as_bytes(seq) -> np.ndarray:
    if isinstance(seq, str):
        return np.frombuffer(seq.encode(), dtype=np.uint8).copy()
    return np.asarray(seq, dtype=np.uint8).copy()


def to_str(arr: np.ndarray) -> str:
    return bytes(arr.tolist()).decode()


def v_real(seq, rng):
    return _as_bytes(seq)


def v_random(seq, rng):
    a = _as_bytes(seq)
    return BASES[rng.integers(0, 4, size=len(a))]


def v_polya(seq, rng):
    a = _as_bytes(seq)
    return np.full(len(a), ord("A"), dtype=np.uint8)


def v_shuffle_mono(seq, rng):
    a = _as_bytes(seq)
    idx = rng.permutation(len(a))
    return a[idx]


def v_shuffle_di(seq, rng):
    """Dinucleotide-preserving shuffle via an Eulerian walk on the dinucleotide graph."""
    a = _as_bytes(seq)
    n = len(a)
    if n < 3:
        return a
    # adjacency: for each base, the list of successors
    succ: Dict[int, List[int]] = {int(b): [] for b in np.unique(a)}
    for i in range(n - 1):
        succ[int(a[i])].append(int(a[i + 1]))
    for k in succ:
        rng.shuffle(succ[k])

    # Ensure connectivity to the final vertex (Altschul-Erikson last-edge trick)
    last = int(a[-1])
    for k in list(succ.keys()):
        if k == last or not succ[k]:
            continue
        # move one edge that leads (eventually) to `last` to the end
        for j, tgt in enumerate(succ[k]):
            if tgt == last:
                succ[k].append(succ[k].pop(j))
                break

    out = [int(a[0])]
    cur = int(a[0])
    for _ in range(n - 1):
        if not succ.get(cur):
            # fall back to a mononucleotide shuffle of the remainder
            rest = a[len(out):]
            out.extend(int(x) for x in rest[rng.permutation(len(rest))])
            break
        nxt = succ[cur].pop(0)
        out.append(nxt)
        cur = nxt
    arr = np.asarray(out[:n], dtype=np.uint8)
    if len(arr) < n:
        arr = np.concatenate([arr, a[len(arr):]])
    return arr


def v_revcomp(seq, rng):
    a = _as_bytes(seq)[::-1]
    return np.asarray([COMPLEMENT.get(int(b), ord("N")) for b in a], dtype=np.uint8)


def make_mask_center(k: int = 200) -> Callable:
    def _f(seq, rng):
        a = _as_bytes(seq)
        mid = len(a) // 2
        lo, hi = mid - k // 2, mid + k // 2
        a[lo:hi] = BASES[rng.integers(0, 4, size=hi - lo)]
        return a
    return _f


REGISTRY: Dict[str, Callable] = {
    "real": v_real,
    "random": v_random,
    "polyA": v_polya,
    "shuffle_mono": v_shuffle_mono,
    "shuffle_di": v_shuffle_di,
    "revcomp": v_revcomp,
    "mask_center_200": make_mask_center(200),
}


def make_variant(name: str, seq, seed: int) -> np.ndarray:
    if name not in REGISTRY:
        raise KeyError(f"unknown variant {name!r}; have {sorted(REGISTRY)}")
    rng = np.random.default_rng(seed)
    return REGISTRY[name](seq, rng)


def substitute(seq, offset: int, new_base: str) -> np.ndarray:
    """Point substitution helper for later ref/alt work."""
    a = _as_bytes(seq)
    a[offset] = ord(new_base)
    return a
