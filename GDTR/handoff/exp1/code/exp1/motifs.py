"""Motif models with provenance, and motif-level annotation.

The rule this module enforces
-----------------------------
**No matrix in this file was typed in by hand, and none can be.**  A :class:`PWM`
cannot be constructed without a :class:`MotifProvenance` saying where its counts
came from, and every count matrix reaching this package arrives by one of three
audited routes:

``derived``
    Counted from real annotated sites in the user's own reference data --
    GENCODE introns on hg38, on chromosomes held out of the analysis panel.
    This is the primary route for splice motifs, because it needs no network,
    is exactly reproducible from two versioned files, and its correctness is
    checkable against a fact nobody has to trust us about: the donor consensus
    must come out GT and the acceptor AG.  :func:`derive_splice_pwms` asserts
    exactly that and refuses to return otherwise.

``jaspar``
    A real JASPAR CORE matrix, loaded from a file or fetched from the JASPAR
    REST API, recorded with its matrix ID, version and SHA-256.

``maxentscan``
    The field-standard splice-strength model of Yeo & Burge (2004), used
    through the ``maxentpy`` implementation.  This package does **not**
    reimplement MaxEntScan; if ``maxentpy`` is absent the scorer raises rather
    than substituting a PWM, because a PWM is a different model (it assumes
    positional independence, which is the assumption MaxEntScan exists to drop).

There is deliberately no fallback and no default motif set.  A missing model is
an error, never a quietly substituted approximation: :func:`annotate_window`
requires a non-empty :class:`MotifSet` and raises without one.

What is a model and what is a measurement
-----------------------------------------
Two kinds of column come out of :func:`annotate_window` and they are never mixed
up.  ``motif_*`` columns are model scores and depend on which motif set was
built.  The columns listed in :data:`MEASUREMENT_COLUMNS` are counts read off
the sequence -- the observed splice dinucleotide, the pyrimidine fraction of the
tract, CpG observed/expected, 3-mer entropy -- and involve no model at all, so
they are available and meaningful whatever motif set is in use.  The
polypyrimidine tract is in the second group on purpose: it is a composition, not
a binding site, and scoring it with a PWM would be inventing a model for
something that can simply be counted.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

BASES = "ACGT"
BASE_IDX = {ord(b): i for i, b in enumerate(BASES)}
COMPLEMENT = {ord("A"): ord("T"), ord("C"): ord("G"),
              ord("G"): ord("C"), ord("T"): ord("A"), ord("N"): ord("N")}

#: Columns that are counts of the sequence, not scores from any model.
MEASUREMENT_COLUMNS = (
    "splice_core",
    "splice_canonical",
    "ppt_fraction",
    "cpg_oe",
    "kmer3_entropy",
)
# ``low_complexity`` used to be here as ``kmer3_entropy < 3.0``. The 3.0 had
# nothing behind it, and a binary derived from a continuous column by an
# unjustified cut is strictly worse than the column: it throws away the ordering
# and adds a decision nobody can defend. Stratify on the entropy itself.

#: Window conventions, in (exonic, intronic) base counts, following the spans
#: Yeo & Burge (2004) defined for the two splice signals.  Kept as named
#: constants because every offset bug in this area comes from writing them twice.
DONOR_EXONIC, DONOR_INTRONIC = 3, 6         # 9 nt total, core GT at index 3
ACCEPTOR_INTRONIC, ACCEPTOR_EXONIC = 20, 3  # 23 nt total, core AG at index 20

BUILDER_VERSION = "exp1-motifs/2"

#: Two thresholds that are *choices*, not facts.  They are named here rather
#: than buried at their use site so that they appear in the manifest and can be
#: swept, since neither is derivable from anything.
# (no tunable constants remain in this module: the former LOW_COMPLEXITY_BITS
#  and HIT_QUANTILE were cuts with no justification and have been removed
#  rather than re-tuned.)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MotifProvenance:
    """Where a count matrix came from.  Required; there is no anonymous PWM."""

    source: str                     # derived | jaspar | maxentscan | file
    detail: str                     # human-readable one-liner
    n_sites: Optional[int] = None   # real sites counted, for source="derived"
    uri: Optional[str] = None       # file path or URL
    sha256: Optional[str] = None    # of the source bytes, when there are any
    reference: Optional[str] = None # GTF/FASTA/assembly versions, or a citation
    built_at: str = ""
    builder: str = BUILDER_VERSION

    def __post_init__(self) -> None:
        if self.source not in ("derived", "jaspar", "maxentscan", "file"):
            raise ValueError(
                f"unknown motif source {self.source!r}; a motif must come from "
                "derived / jaspar / maxentscan / file"
            )
        if not self.detail:
            raise ValueError("provenance.detail must say where this matrix came from")
        if not self.built_at:
            object.__setattr__(self, "built_at",
                               datetime.now(timezone.utc).isoformat(timespec="seconds"))


# --------------------------------------------------------------------------
# PWM
# --------------------------------------------------------------------------

@dataclass
class PWM:
    """A position weight matrix, stored as the *counts* it was built from.

    Counts rather than probabilities, because the count matrix is the auditable
    object: it carries the sample size, it makes the pseudo-count explicit
    instead of baked in, and it hashes to something a reader can check.
    """

    name: str
    counts: np.ndarray              # [L, 4] non-negative, columns A C G T
    provenance: MotifProvenance
    pseudocount: float = 0.25       # Jeffreys-style; recorded, never implicit
    core_offset: int = 0            # index of the functional core in the motif
    core_len: int = 2

    def __post_init__(self) -> None:
        m = np.asarray(self.counts, dtype=np.float64)
        if m.ndim != 2 or m.shape[1] != 4:
            raise ValueError(f"{self.name}: counts must be [L, 4], got {m.shape}")
        if m.shape[0] == 0:
            raise ValueError(f"{self.name}: empty matrix")
        if not np.isfinite(m).all() or (m < 0).any():
            raise ValueError(f"{self.name}: counts must be finite and non-negative")
        if self.pseudocount <= 0:
            raise ValueError(f"{self.name}: pseudocount must be > 0")
        if not (0 <= self.core_offset < m.shape[0]):
            raise ValueError(f"{self.name}: core_offset outside the motif")
        self.counts = m

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_counts(cls, name: str, counts, provenance: MotifProvenance, **kw) -> "PWM":
        return cls(name=name, counts=np.asarray(counts, dtype=np.float64),
                   provenance=provenance, **kw)

    # -- derived quantities -------------------------------------------------

    @property
    def length(self) -> int:
        return self.counts.shape[0]

    @property
    def n_observations(self) -> float:
        """Sites behind the matrix, read off the counts themselves."""
        return float(self.counts.sum(axis=1).max())

    def probs(self) -> np.ndarray:
        m = self.counts + self.pseudocount
        return m / m.sum(axis=1, keepdims=True)

    def log_odds(self, background: Optional[np.ndarray] = None) -> np.ndarray:
        bg = np.asarray(background if background is not None else np.full(4, 0.25),
                        dtype=np.float64)
        if bg.shape != (4,) or not np.isfinite(bg).all() or (bg < 0).any():
            raise ValueError("background must be 4 finite non-negative numbers")
        # A degenerate background (poly-A, or an all-one-base window) would
        # otherwise divide by zero and produce silent +/-inf scores.
        bg = np.clip(bg, 1e-3, None)
        bg = bg / bg.sum()
        return np.log2(self.probs() / bg[None, :])

    def information_content(self, background: Optional[np.ndarray] = None) -> np.ndarray:
        """Per-position relative entropy to the background, in bits.

        Relative entropy rather than ``2 - H`` because the background here is a
        measured genomic composition, not uniform.
        """
        p = self.probs()
        lo = self.log_odds(background)
        return (p * lo).sum(axis=1)

    def consensus(self) -> str:
        return "".join(BASES[i] for i in self.counts.argmax(axis=1))

    def digest(self) -> str:
        """SHA-256 of the integer counts -- the matrix's identity."""
        payload = json.dumps(
            dict(name=self.name,
                 counts=np.rint(self.counts).astype(np.int64).tolist(),
                 core_offset=self.core_offset, core_len=self.core_len),
            sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> Dict[str, object]:
        return dict(name=self.name,
                    counts=self.counts.tolist(),
                    pseudocount=self.pseudocount,
                    core_offset=self.core_offset,
                    core_len=self.core_len,
                    consensus=self.consensus(),
                    n_observations=self.n_observations,
                    digest=self.digest(),
                    provenance=asdict(self.provenance))

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "PWM":
        prov = MotifProvenance(**d["provenance"])          # type: ignore[arg-type]
        pwm = cls(name=str(d["name"]),
                  counts=np.asarray(d["counts"], dtype=np.float64),
                  provenance=prov,
                  pseudocount=float(d.get("pseudocount", 0.25)),
                  core_offset=int(d.get("core_offset", 0)),
                  core_len=int(d.get("core_len", 2)))
        recorded = d.get("digest")
        if recorded and recorded != pwm.digest():
            raise ValueError(
                f"{pwm.name}: matrix digest does not match the manifest "
                f"({recorded} vs {pwm.digest()}); the file has been edited"
            )
        return pwm


# --------------------------------------------------------------------------
# motif set
# --------------------------------------------------------------------------

@dataclass
class MotifSet:
    """The motif models in use, their background, and how to prove it."""

    pwms: Dict[str, PWM] = field(default_factory=dict)
    background: Optional[np.ndarray] = None   # measured; None = per-window
    note: str = ""

    def __bool__(self) -> bool:
        return bool(self.pwms)

    def __len__(self) -> int:
        return len(self.pwms)

    def require(self, what: str = "this analysis") -> "MotifSet":
        if not self.pwms:
            raise ValueError(
                f"no motif models loaded, so {what} cannot run. Build one with "
                "`python -m exp1 step0-motifs build ...` (derives donor/acceptor "
                "from GENCODE on held-out chromosomes) or load real matrices with "
                "load_jaspar(). This package ships no built-in matrices on purpose."
            )
        return self

    def add(self, pwm: PWM) -> "MotifSet":
        if pwm.name in self.pwms:
            raise ValueError(f"duplicate motif name {pwm.name!r}")
        self.pwms[pwm.name] = pwm
        return self

    def model_columns(self) -> Tuple[str, ...]:
        cols: List[str] = []
        for name in self.pwms:
            cols += [f"motif_{name}_score", f"motif_{name}_dist"]
        return tuple(cols) + ("motif_best", "motif_best_score")

    def to_manifest(self) -> Dict[str, object]:
        return dict(
            builder=BUILDER_VERSION,
            written_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            note=self.note,
            background=None if self.background is None else np.asarray(self.background).tolist(),
            measurement_columns=list(MEASUREMENT_COLUMNS),
            model_columns=list(self.model_columns()),
            motifs=[p.to_dict() for p in self.pwms.values()],
        )

    def save(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_manifest(), indent=2))
        return str(p)

    @classmethod
    def load(cls, path: str) -> "MotifSet":
        man = json.loads(Path(path).read_text())
        bg = man.get("background")
        ms = cls(background=None if bg is None else np.asarray(bg, dtype=np.float64),
                 note=str(man.get("note", "")))
        for d in man.get("motifs", []):
            ms.add(PWM.from_dict(d))
        return ms


# --------------------------------------------------------------------------
# route 1: derive splice PWMs from real annotated sites
# --------------------------------------------------------------------------

def iter_introns(gtf_path: str, chrom: str) -> Iterable[Tuple[int, int, str]]:
    """Unique (start, end, strand) introns of ``chrom``, from exon records.

    Introns are de-duplicated across transcripts: the same intron shared by ten
    isoforms is one site, not ten, or the count matrix silently weights by how
    well-annotated a gene is.
    """
    from .labels import _iter_gtf

    exons_by_tx: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    strand_by_tx: Dict[str, str] = {}
    for feat, s, e, strand, tid, _ in _iter_gtf(gtf_path, chrom):
        if feat != "exon" or not tid or e <= s:
            continue
        exons_by_tx[tid].append((s, e))
        strand_by_tx[tid] = strand

    seen = set()
    for tid, exons in exons_by_tx.items():
        if len(exons) < 2:
            continue
        strand = strand_by_tx.get(tid, "+")
        exons = sorted(set(exons))
        for (_s1, e1), (s2, _e2) in zip(exons[:-1], exons[1:]):
            if s2 - e1 < 4:
                continue
            key = (e1, s2, strand)
            if key in seen:
                continue
            seen.add(key)
            yield key


def _revcomp_bytes(arr: np.ndarray) -> np.ndarray:
    return np.asarray([COMPLEMENT.get(int(b), ord("N")) for b in arr[::-1]],
                      dtype=np.uint8)


def _site_windows(seq: np.ndarray, introns: Iterable[Tuple[int, int, str]]
                  ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Donor and acceptor windows in *transcript* orientation.

    The strand handling here is the part that is easy to get wrong and expensive
    to get wrong quietly -- the companion manuscript records a whole appendix on
    exactly this class of bug.  It is checked downstream by asserting the
    recovered consensus, which fails loudly if any of these four slices is off.
    """
    dlen = DONOR_EXONIC + DONOR_INTRONIC
    alen = ACCEPTOR_INTRONIC + ACCEPTOR_EXONIC
    donors: List[np.ndarray] = []
    acceptors: List[np.ndarray] = []
    n = len(seq)

    for s, e, strand in introns:
        if strand == "+":
            d0, d1 = s - DONOR_EXONIC, s + DONOR_INTRONIC
            a0, a1 = e - ACCEPTOR_INTRONIC, e + ACCEPTOR_EXONIC
            if d0 >= 0 and d1 <= n:
                donors.append(seq[d0:d1])
            if a0 >= 0 and a1 <= n:
                acceptors.append(seq[a0:a1])
        else:
            # transcript runs right-to-left: donor sits at the intron's right
            # end, acceptor at its left end, both read reverse-complemented.
            d0, d1 = e - DONOR_INTRONIC, e + DONOR_EXONIC
            a0, a1 = s - ACCEPTOR_EXONIC, s + ACCEPTOR_INTRONIC
            if d0 >= 0 and d1 <= n:
                donors.append(_revcomp_bytes(seq[d0:d1]))
            if a0 >= 0 and a1 <= n:
                acceptors.append(_revcomp_bytes(seq[a0:a1]))

    donors = [w for w in donors if len(w) == dlen]
    acceptors = [w for w in acceptors if len(w) == alen]
    return donors, acceptors


def count_matrix(windows: Sequence[np.ndarray], width: int) -> Tuple[np.ndarray, int]:
    """Stack windows into an [width, 4] count matrix, dropping any with N."""
    counts = np.zeros((width, 4), dtype=np.int64)
    used = 0
    for w in windows:
        idx = encode(w)
        if (idx == 255).any():
            continue
        counts[np.arange(width), idx] += 1
        used += 1
    return counts, used


def derive_splice_pwms(
    fasta: str,
    gtf: str,
    chroms: Sequence[str],
    panel_chroms: Sequence[str] = (),
    canonical_only: bool = True,
    min_sites: int = 500,
) -> Tuple[MotifSet, Dict[str, object]]:
    """Count donor and acceptor matrices from real GENCODE introns.

    ``chroms`` must be disjoint from ``panel_chroms``.  Deriving a motif model
    on the same chromosomes the effects are measured on would let the model see
    its own test set, which is the mistake the companion manuscript's Limitations
    section is about; so it is refused here rather than warned about.

    ``canonical_only`` restricts the counts to GT-AG introns.  That is not a
    cosmetic filter: it makes the resulting matrix a model of the U2 signal
    rather than a mixture of U2, U12 and annotation noise, and the non-canonical
    sites it excludes remain available downstream as their own stratum via the
    observed ``splice_core`` measurement.

    Returns the motif set and a report -- site counts, per-position information
    content and recovered consensus -- meant to be printed and pasted into the
    methods section.
    """
    overlap = sorted(set(chroms) & set(panel_chroms))
    if overlap:
        raise ValueError(
            f"refusing to derive motifs on {overlap}: those chromosomes carry the "
            "analysis panel, so the motif model would be fit on the data it is "
            "later used to stratify. Pass held-out chromosomes."
        )
    if not chroms:
        raise ValueError("no chromosomes given to derive motifs from")

    from pyfaidx import Fasta

    fa = Fasta(fasta, as_raw=True, sequence_always_upper=True)
    dlen = DONOR_EXONIC + DONOR_INTRONIC
    alen = ACCEPTOR_INTRONIC + ACCEPTOR_EXONIC

    all_donor: List[np.ndarray] = []
    all_accept: List[np.ndarray] = []
    per_chrom: Dict[str, Dict[str, int]] = {}
    base_counts = np.zeros(4, dtype=np.int64)

    for chrom in chroms:
        seq = np.frombuffer(str(fa[chrom][:]).upper().encode(), dtype=np.uint8)
        idx = encode(seq)
        for i in range(4):
            base_counts[i] += int((idx == i).sum())
        introns = list(iter_introns(gtf, chrom))
        d, a = _site_windows(seq, introns)
        if canonical_only:
            d = [w for w in d if bytes(w[DONOR_EXONIC:DONOR_EXONIC + 2]) == b"GT"]
            a = [w for w in a if
                 bytes(w[ACCEPTOR_INTRONIC - 2:ACCEPTOR_INTRONIC]) == b"AG"]
        per_chrom[chrom] = dict(introns=len(introns), donors=len(d), acceptors=len(a))
        all_donor += d
        all_accept += a

    d_counts, n_d = count_matrix(all_donor, dlen)
    a_counts, n_a = count_matrix(all_accept, alen)
    if n_d < min_sites or n_a < min_sites:
        raise ValueError(
            f"only {n_d} donor and {n_a} acceptor sites usable (need {min_sites}); "
            "check the GTF chromosome naming matches the FASTA"
        )

    background = (base_counts + 1.0) / (base_counts + 1.0).sum()
    ref = f"GENCODE={Path(gtf).name}; assembly={Path(fasta).name}; chroms={','.join(chroms)}"
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    donor = PWM.from_counts(
        "splice_donor_U2", d_counts,
        MotifProvenance(source="derived",
                        detail=("5' splice site counted from GENCODE introns, "
                                f"{DONOR_EXONIC} exonic + {DONOR_INTRONIC} intronic nt"
                                + (", GT-AG only" if canonical_only else "")),
                        n_sites=n_d, uri=str(gtf), reference=ref, built_at=stamp),
        core_offset=DONOR_EXONIC, core_len=2)
    acceptor = PWM.from_counts(
        "splice_acceptor_U2", a_counts,
        MotifProvenance(source="derived",
                        detail=("3' splice site counted from GENCODE introns, "
                                f"{ACCEPTOR_INTRONIC} intronic + {ACCEPTOR_EXONIC} exonic nt"
                                + (", GT-AG only" if canonical_only else "")),
                        n_sites=n_a, uri=str(gtf), reference=ref, built_at=stamp),
        core_offset=ACCEPTOR_INTRONIC - 2, core_len=2)

    # The check that makes this route trustworthy without trusting us: if any of
    # the four strand-dependent slices above is off by one, or a strand is
    # handled backwards, the consensus stops reading GT / AG and we stop here.
    d_core = donor.consensus()[DONOR_EXONIC:DONOR_EXONIC + 2]
    a_core = acceptor.consensus()[ACCEPTOR_INTRONIC - 2:ACCEPTOR_INTRONIC]
    if d_core != "GT" or a_core != "AG":
        raise AssertionError(
            f"derived consensus is {d_core!r}/{a_core!r}, expected 'GT'/'AG'. "
            "The site windows are misaligned -- do not use this matrix."
        )

    ms = MotifSet(background=background,
                  note=f"splice motifs derived from {','.join(chroms)}")
    ms.add(donor).add(acceptor)

    report = dict(
        chroms=list(chroms), panel_chroms=list(panel_chroms),
        canonical_only=canonical_only, per_chrom=per_chrom,
        background=dict(zip(BASES, background.round(4).tolist())),
        donor=dict(n_sites=n_d, consensus=donor.consensus(),
                   ic_bits=donor.information_content(background).round(3).tolist(),
                   total_ic=float(donor.information_content(background).sum())),
        acceptor=dict(n_sites=n_a, consensus=acceptor.consensus(),
                      ic_bits=acceptor.information_content(background).round(3).tolist(),
                      total_ic=float(acceptor.information_content(background).sum())),
    )
    return ms, report


# --------------------------------------------------------------------------
# route 2: real JASPAR matrices
# --------------------------------------------------------------------------

def load_jaspar(path: str, core_offset: int = 0, core_len: int = 2) -> List[PWM]:
    """Parse a JASPAR ``.jaspar`` / ``.pfm`` file of integer counts.

    Strict on purpose: four rows, equal widths, non-negative integers.  A file
    that does not parse is an error, not a matrix with some rows dropped.
    """
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    lines = raw.decode().splitlines()

    out: List[PWM] = []
    name = Path(path).stem
    rows: List[List[float]] = []

    def flush() -> None:
        if not rows:
            return
        if len(rows) != 4:
            raise ValueError(f"{path}: expected 4 rows (A C G T), got {len(rows)}")
        widths = {len(r) for r in rows}
        if len(widths) != 1:
            raise ValueError(f"{path}: rows have unequal widths {sorted(widths)}")
        counts = np.asarray(rows, dtype=np.float64).T
        out.append(PWM.from_counts(
            name, counts,
            MotifProvenance(source="jaspar", detail=f"JASPAR matrix {name}",
                            uri=str(path), sha256=sha,
                            n_sites=int(round(counts.sum(axis=1).max()))),
            core_offset=min(core_offset, counts.shape[0] - 1), core_len=core_len))
        rows.clear()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            parts = line[1:].split()
            name = parts[0] if parts else name
            if len(parts) > 1:
                name = f"{parts[0]}_{parts[1]}"
            continue
        body = line.split("[", 1)[-1].split("]", 1)[0] if "[" in line else line
        toks = [t for t in body.replace("A", " ").replace("C", " ")
                .replace("G", " ").replace("T", " ").split() if _is_num(t)]
        if toks:
            rows.append([float(t) for t in toks])
    flush()
    if not out:
        raise ValueError(f"{path}: no matrices parsed")
    return out


JASPAR_API = "https://jaspar.elixir.no/api/v1/matrix/{mid}.jaspar"


def fetch_jaspar(matrix_ids: Sequence[str], outdir: str) -> Dict[str, Dict[str, str]]:
    """Download JASPAR matrices and record what was actually fetched.

    An unversioned id (``MA0108``) is allowed and resolves to the collection's
    current version; the resolved id is read back out of the file's own header
    and recorded, so the manifest pins it from then on.  That is different from
    guessing a version number: nothing here decides which version is current,
    JASPAR does, and what it returned is written down.

    A pinned id (``MA0108.3``) is verified against the header and a mismatch is
    an error.
    """
    import urllib.request

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Dict[str, str]] = {}
    for mid in matrix_ids:
        url = JASPAR_API.format(mid=mid)
        with urllib.request.urlopen(url, timeout=60) as fh:   # noqa: S310
            body = fh.read()
        if not body.strip().startswith(b">"):
            raise ValueError(f"{url}: response is not a JASPAR matrix")
        header = body.decode(errors="replace").splitlines()[0]
        resolved = header[1:].split()[0]
        if "." in mid and resolved != mid:
            raise ValueError(
                f"asked for {mid} but JASPAR returned {resolved}; refusing to "
                "record one id for another matrix"
            )
        dest = out / f"{resolved}.jaspar"
        dest.write_bytes(body)
        manifest[mid] = dict(path=str(dest), url=url, resolved_id=resolved,
                             requested_id=mid,
                             sha256=hashlib.sha256(body).hexdigest())
    return manifest


# --------------------------------------------------------------------------
# route 3: MaxEntScan, through a verified implementation only
# --------------------------------------------------------------------------

def maxentscan_scorer():
    """Return ``(score5, score3)`` from ``maxentpy``, or raise.

    MaxEntScan is the field standard for splice-site *strength* precisely
    because it models dependencies between positions, which a PWM cannot.  This
    package therefore does not approximate it with one, and does not reimplement
    it either -- an unverified reimplementation of a published scoring table is
    exactly the kind of number that cannot be defended in review.
    """
    try:
        from maxentpy import maxent                      # type: ignore
        from maxentpy.maxent import load_matrix5, load_matrix3  # type: ignore
    except ImportError as exc:                           # pragma: no cover
        raise ImportError(
            "MaxEntScan scoring needs the `maxentpy` package (pip install maxentpy). "
            "This package will not substitute a PWM for it: a PWM assumes the "
            "positions are independent, which is the assumption MaxEntScan drops."
        ) from exc
    m5, m3 = load_matrix5(), load_matrix3()
    return (lambda s: maxent.score5(s, matrix=m5),
            lambda s: maxent.score3(s, matrix=m3))


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------

def _is_num(t: str) -> bool:
    try:
        float(t)
        return True
    except ValueError:
        return False


def encode(seq_upper: np.ndarray) -> np.ndarray:
    """ASCII bases -> index array with 255 for anything that is not ACGT."""
    out = np.full(len(seq_upper), 255, dtype=np.uint8)
    for code, idx in BASE_IDX.items():
        out[seq_upper == code] = idx
    return out


def revcomp(seq_upper: np.ndarray) -> np.ndarray:
    return _revcomp_bytes(seq_upper)


def background_from(seq_upper: np.ndarray) -> np.ndarray:
    """Mononucleotide composition of a window, with a pseudo-count."""
    idx = encode(seq_upper)
    counts = np.array([(idx == i).sum() for i in range(4)], dtype=np.float64)
    if counts.sum() == 0:
        return np.full(4, 0.25)
    counts = counts + 1.0
    return counts / counts.sum()


def scan_pwm(seq_upper: np.ndarray, pwm: PWM,
             background: Optional[np.ndarray] = None) -> np.ndarray:
    """Log-odds score of ``pwm`` starting at every offset.  NaN where N occurs."""
    idx = encode(seq_upper)
    lo = pwm.log_odds(background)
    n, L = len(idx), pwm.length
    if n < L:
        return np.full(max(n, 0), np.nan)
    scores = np.zeros(n - L + 1, dtype=np.float64)
    bad = np.zeros(n - L + 1, dtype=bool)
    for k in range(L):
        col = idx[k : k + n - L + 1]
        valid = col != 255
        safe = np.where(valid, col, 0)
        scores += np.where(valid, lo[k, safe], 0.0)
        bad |= ~valid
    scores[bad] = np.nan
    return np.concatenate([scores, np.full(L - 1, np.nan)])


# --------------------------------------------------------------------------
# per-position annotation
# --------------------------------------------------------------------------

@dataclass
class MotifAnnotation:
    """Per-position motif columns for one window."""

    columns: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, np.ndarray]:
        return self.columns


def annotate_window(
    seq_upper: np.ndarray,
    scored: slice,
    context: np.ndarray,
    motif_set: MotifSet,
    both_strands: bool = True,
) -> MotifAnnotation:
    """Build motif columns for the scored region of one window.

    ``motif_set`` is required and must be non-empty; there is no default set to
    fall back on.  Model columns (``motif_*``) come from that set; the columns
    in :data:`MEASUREMENT_COLUMNS` are counts of the sequence and are produced
    either way.

    Per position:
      ``motif_<name>_score``   best log-odds of a hit covering this position
      ``motif_<name>_dist``    signed distance to that hit's core
      ``motif_best`` / ``motif_best_score``
      ``splice_core``          observed 2-nt core where the context says
                               splice_donor / splice_acceptor
      ``splice_canonical``     1.0 for GT (donor) / AG (acceptor), else 0.0
      ``ppt_fraction``         pyrimidine fraction just upstream (measurement)
      ``cpg_oe``, ``kmer3_entropy``
    """
    if not isinstance(motif_set, MotifSet):
        raise TypeError("annotate_window needs a MotifSet; see step0-motifs")
    motif_set.require("motif annotation")

    bg = motif_set.background if motif_set.background is not None \
        else background_from(seq_upper)
    n_scored = scored.stop - scored.start
    cols: Dict[str, np.ndarray] = {}

    best_score = np.full(n_scored, -np.inf)
    best_name = np.array(["none"] * n_scored, dtype=object)

    for name, pwm in motif_set.pwms.items():
        fwd = scan_pwm(seq_upper, pwm, bg)
        scores = fwd
        if both_strands:
            rc = scan_pwm(revcomp(seq_upper), pwm, bg)[::-1]
            rc = np.roll(rc, -(pwm.length - 1))
            f = np.where(np.isnan(fwd), -np.inf, fwd)
            r = np.where(np.isnan(rc), -np.inf, rc)
            scores = np.fmax(f, r)
        cover = _max_over_coverage(scores, pwm.length)
        dist = _dist_to_best_core(scores, pwm)
        cover = np.where(np.isfinite(cover), cover, np.nan)
        cols[f"motif_{name}_score"] = cover[scored].astype(np.float32)
        cols[f"motif_{name}_dist"] = dist[scored].astype(np.float32)
        here = np.where(np.isnan(cover[scored]), -np.inf, cover[scored])
        better = here > best_score
        best_score = np.where(better, here, best_score)
        best_name = np.where(better, name, best_name)

    cols["motif_best"] = best_name
    cols["motif_best_score"] = np.where(np.isfinite(best_score),
                                        best_score, np.nan).astype(np.float32)

    cols.update(measure_window(seq_upper, scored, context))
    return MotifAnnotation(columns=cols)


def measure_window(seq_upper: np.ndarray, scored: slice,
                   context: np.ndarray) -> Dict[str, np.ndarray]:
    """The :data:`MEASUREMENT_COLUMNS`: counts of the sequence, no model.

    Separated from :func:`annotate_window` so that these are available with no
    motif set loaded at all.  They cannot be wrong because a matrix was wrong;
    they can only be wrong because the sequence is.
    """
    cols: Dict[str, np.ndarray] = {}
    core, canon = _splice_cores(seq_upper, scored, context)
    cols["splice_core"] = core
    cols["splice_canonical"] = canon
    cols["ppt_fraction"] = _ppt_fraction(seq_upper)[scored].astype(np.float32)
    cols["cpg_oe"] = _cpg_obs_exp(seq_upper)[scored].astype(np.float32)
    cols["kmer3_entropy"] = _kmer_entropy(seq_upper, k=3)[scored].astype(np.float32)
    assert set(cols) == set(MEASUREMENT_COLUMNS), "MEASUREMENT_COLUMNS is out of date"
    return cols


def _max_over_coverage(start_scores: np.ndarray, length: int) -> np.ndarray:
    """For each base, the best score among motif instances covering it."""
    n = len(start_scores)
    s = np.where(np.isnan(start_scores), -np.inf, start_scores)
    out = np.full(n, -np.inf)
    for off in range(length):
        shifted = np.full(n, -np.inf)
        if off == 0:
            shifted[:] = s
        else:
            shifted[off:] = s[:-off]
        out = np.fmax(out, shifted)
    return out


def _dist_to_best_core(start_scores: np.ndarray, pwm: PWM) -> np.ndarray:
    """Signed distance from each base to the core of the window's BEST hit.

    The previous definition called the top 1% of scores *within each window*
    "hits", which fixes the hit density at 1% whether or not the window contains
    a motif, and makes the column incomparable between windows. There is no
    threshold-free notion of "a strong hit", but there is a threshold-free
    notion of "the best one here", so that is what this measures. How good that
    best hit actually is lives in the score column beside it, and a called site
    is decided in :mod:`exp1.sitedefs` against a composition-matched null.
    """
    s = np.where(np.isnan(start_scores), -np.inf, start_scores)
    if not np.isfinite(s).any():
        return np.full(len(s), np.nan)
    hits = np.array([int(np.nanargmax(np.where(np.isfinite(s), s, -np.inf)))]) + pwm.core_offset
    if hits.size == 0:
        return np.full(len(s), np.nan)
    pos = np.arange(len(s))
    j = np.searchsorted(hits, pos)
    left = np.where(j > 0, pos - hits[np.clip(j - 1, 0, len(hits) - 1)], 10**6)
    right = np.where(j < len(hits), hits[np.clip(j, 0, len(hits) - 1)] - pos, 10**6)
    signed = np.where(np.abs(left) <= np.abs(right), -left, right)
    return signed.astype(np.float64)


# --------------------------------------------------------------------------
# measurements (no model involved)
# --------------------------------------------------------------------------

def _splice_cores(seq_upper: np.ndarray, scored: slice,
                  context: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Observed dinucleotide at annotated splice positions, and canonicality."""
    n = scored.stop - scored.start
    core = np.array([""] * n, dtype=object)
    canon = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        ctx = context[i]
        if ctx not in ("splice_donor", "splice_acceptor"):
            continue
        g = scored.start + i
        if g + 2 > len(seq_upper):
            continue
        di = bytes(seq_upper[g : g + 2].tolist()).decode(errors="ignore")
        core[i] = di
        if ctx == "splice_donor":
            canon[i] = 1.0 if di == "GT" else 0.0
        else:
            canon[i] = 1.0 if di == "AG" else 0.0
    return core, canon


def _ppt_fraction(seq_upper: np.ndarray, span: int = 20, gap: int = 5) -> np.ndarray:
    """Pyrimidine (C/T) fraction in the ``span`` bases ending ``gap`` upstream.

    The polypyrimidine tract is a composition, not a binding site, so it is
    counted rather than scored with a matrix.  The window follows the tract's
    usual position relative to a 3' splice site.
    """
    pyr = ((seq_upper == ord("C")) | (seq_upper == ord("T"))).astype(np.float64)
    known = ((seq_upper == ord("A")) | (seq_upper == ord("C")) |
             (seq_upper == ord("G")) | (seq_upper == ord("T"))).astype(np.float64)
    cs_p = np.concatenate([[0.0], np.cumsum(pyr)])
    cs_k = np.concatenate([[0.0], np.cumsum(known)])
    n = len(seq_upper)
    pos = np.arange(n)
    hi = np.clip(pos - gap, 0, n)
    lo = np.clip(hi - span, 0, n)
    num = cs_p[hi] - cs_p[lo]
    den = cs_k[hi] - cs_k[lo]
    out = np.full(n, np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _cpg_obs_exp(seq_upper: np.ndarray, half: int = 100) -> np.ndarray:
    c = (seq_upper == ord("C")).astype(np.float64)
    g = (seq_upper == ord("G")).astype(np.float64)
    cg = np.zeros(len(seq_upper))
    cg[:-1] = c[:-1] * g[1:]
    k = 2 * half + 1

    def roll(x):
        cs = np.concatenate([[0.0], np.cumsum(x)])
        lo = np.clip(np.arange(len(x)) - half, 0, len(x))
        hi = np.clip(np.arange(len(x)) + half + 1, 0, len(x))
        return cs[hi] - cs[lo]

    nc, ng, ncg = roll(c), roll(g), roll(cg)
    denom = np.maximum(nc * ng, 1.0)
    return (ncg * k) / denom


def _kmer_entropy(seq_upper: np.ndarray, k: int = 3, half: int = 50) -> np.ndarray:
    """Shannon entropy of k-mer usage in a sliding window, in bits."""
    idx = encode(seq_upper).astype(np.int32)
    n = len(idx)
    codes = np.full(n, -1, dtype=np.int32)
    valid = np.ones(n, dtype=bool)
    acc = np.zeros(n, dtype=np.int32)
    for j in range(k):
        sl = idx[j : n - k + 1 + j]
        valid[: n - k + 1] &= sl != 255
        acc[: n - k + 1] = acc[: n - k + 1] * 4 + np.where(sl == 255, 0, sl)
    codes[: n - k + 1] = np.where(valid[: n - k + 1], acc[: n - k + 1], -1)

    out = np.zeros(n)
    win = 2 * half + 1
    for start in range(0, n, win):
        stop = min(start + win, n)
        seg = codes[start:stop]
        seg = seg[seg >= 0]
        if seg.size == 0:
            continue
        _, cnt = np.unique(seg, return_counts=True)
        p = cnt / cnt.sum()
        out[start:stop] = float(-(p * np.log2(p)).sum())
    return out


def strength_quartiles(score: np.ndarray,
                       labels: Sequence[str] = ("q1", "q2", "q3", "q4")) -> np.ndarray:
    """Within-panel quartile label of a motif score, NaN-safe."""
    s = np.asarray(score, dtype=np.float64)
    ok = np.isfinite(s)
    out = np.array(["na"] * len(s), dtype=object)
    if ok.sum() < 8:
        return out
    qs = np.nanquantile(s[ok], [0.25, 0.5, 0.75])
    binned = np.digitize(s[ok], qs)
    out[ok] = [labels[int(b)] for b in binned]
    return out
