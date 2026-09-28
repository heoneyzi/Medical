"""Motif perturbation plans -- the interventional half of the motif work.

This reproduces the perturbation design the earlier work used (core edit vs
flank shuffle) and generalises it, so that a single extraction pass can carry
paired sequences for necessity, sufficiency and dose-response.

Families
--------
core_mut      mutate the functional core, keep the flanks        (necessity)
flank_shuffle dinucleotide-preserving shuffle of the flanks,
              keep the core                                       (context test)
both          core mutated *and* flanks shuffled                  (interaction)
scramble      shuffle the motif itself, keep composition          (composition control)
insert        plant a motif into a matched background             (sufficiency)
rescue        insert-after-disrupt at the original site           (rescue)
spacing       insert two copies at a swept spacing                (grammar)

Each perturbation emits a row of metadata alongside the sequence, so that the
extraction table carries ``pert_family``, ``pert_site``, ``pert_param`` and a
``pert_pair_id`` linking an edit to its own control.  Paired comparisons are then
within-window *and* within-site, which is what makes the effect estimate a
difference rather than a between-locus comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .motifs import BASES, MotifSet, PWM
from .variants import _as_bytes, to_str, v_shuffle_di

# Knockout edits.  These are deliberate *choices*, not facts about biology: each
# replaces a splice core with a dinucleotide that is not any splice core, so the
# edit cannot accidentally convert one class into another.
CORE_MUTATIONS = {
    "GT": "AA",   # U2 donor core knockout, as used in the earlier work
    "GC": "AA",   # GC-AG donor
    "AT": "GG",   # AT-AC donor
    "AG": "TT",   # acceptor core knockout
    "AC": "TT",   # AT-AC acceptor
    "A": "G",     # branch adenine
}


@dataclass
class PerturbSite:
    """One locus to perturb inside a window."""

    offset: int                 # start of the motif within the window
    motif: str                  # PWM name
    core_offset: int = 0        # core start relative to ``offset``
    core_len: int = 2
    flank: int = 100            # +- bases treated as the flank
    label: str = ""             # e.g. the annotation context at the core


@dataclass
class PerturbedSeq:
    seq: np.ndarray
    meta: Dict[str, object]


def _dinuc_shuffle(seg: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return v_shuffle_di(seg, rng)


def build_perturbations(
    window_seq,
    sites: Sequence[PerturbSite],
    motif_set: MotifSet,
    families: Sequence[str] = ("core_mut", "flank_shuffle"),
    seed: int = 0,
    spacings: Sequence[int] = (10, 20, 40, 80),
    copies: Sequence[int] = (1, 2, 4),
) -> List[PerturbedSeq]:
    """Return the reference plus one sequence per (site, family, parameter)."""
    motif_set.require("perturbation building")
    ref = _as_bytes(window_seq)
    out: List[PerturbedSeq] = [PerturbedSeq(
        seq=ref.copy(),
        meta=dict(pert_family="reference", pert_site=-1, pert_motif="",
                  pert_param="", pert_pair_id="ref"),
    )]

    for si, site in enumerate(sites):
        pair = f"s{si}"
        cs = site.offset + site.core_offset
        ce = cs + site.core_len
        core_seq = to_str(ref[cs:ce])

        if "core_mut" in families:
            mut = ref.copy()
            repl = CORE_MUTATIONS.get(core_seq, "".join(
                "A" if chr(b) != "A" else "C" for b in ref[cs:ce]))
            mut[cs:ce] = _as_bytes(repl)[: ce - cs]
            out.append(PerturbedSeq(mut, dict(
                pert_family="core_mut", pert_site=si, pert_motif=site.motif,
                pert_param=f"{core_seq}->{repl}", pert_pair_id=pair,
                pert_core_offset=cs, pert_label=site.label)))

        if "flank_shuffle" in families:
            sh = ref.copy()
            lo = max(cs - site.flank, 0)
            hi = min(ce + site.flank, len(ref))
            left = _dinuc_shuffle(ref[lo:cs], seed + 13 * si)
            right = _dinuc_shuffle(ref[ce:hi], seed + 17 * si)
            sh[lo:cs] = left
            sh[ce:hi] = right
            out.append(PerturbedSeq(sh, dict(
                pert_family="flank_shuffle", pert_site=si, pert_motif=site.motif,
                pert_param=f"flank{site.flank}", pert_pair_id=pair,
                pert_core_offset=cs, pert_label=site.label)))

        if "both" in families:
            bo = ref.copy()
            lo, hi = max(cs - site.flank, 0), min(ce + site.flank, len(ref))
            bo[lo:cs] = _dinuc_shuffle(ref[lo:cs], seed + 23 * si)
            bo[ce:hi] = _dinuc_shuffle(ref[ce:hi], seed + 29 * si)
            repl = CORE_MUTATIONS.get(core_seq, "AA")
            bo[cs:ce] = _as_bytes(repl)[: ce - cs]
            out.append(PerturbedSeq(bo, dict(
                pert_family="both", pert_site=si, pert_motif=site.motif,
                pert_param="core+flank", pert_pair_id=pair,
                pert_core_offset=cs, pert_label=site.label)))

        if "scramble" in families:
            sc = ref.copy()
            rng = np.random.default_rng(seed + 31 * si)
            lo, hi = site.offset, site.offset + _motif_len(site, motif_set)
            seg = ref[lo:hi].copy()
            rng.shuffle(seg)
            sc[lo:hi] = seg
            out.append(PerturbedSeq(sc, dict(
                pert_family="scramble", pert_site=si, pert_motif=site.motif,
                pert_param="composition_preserved", pert_pair_id=pair,
                pert_core_offset=cs, pert_label=site.label)))

        if "rescue" in families:
            rs = ref.copy()
            lo, hi = max(cs - site.flank, 0), min(ce + site.flank, len(ref))
            rs[lo:cs] = _dinuc_shuffle(ref[lo:cs], seed + 13 * si)   # same shuffle as above
            rs[ce:hi] = _dinuc_shuffle(ref[ce:hi], seed + 17 * si)
            rs[cs:ce] = _as_bytes(core_seq)                            # core restored
            out.append(PerturbedSeq(rs, dict(
                pert_family="rescue", pert_site=si, pert_motif=site.motif,
                pert_param="core_restored", pert_pair_id=pair,
                pert_core_offset=cs, pert_label=site.label)))

    if "insert" in families or "spacing" in families:
        out.extend(_insertion_series(ref, seed, spacings, copies, motif_set,
                                     do_insert="insert" in families,
                                     do_spacing="spacing" in families))
    return out


def _motif_len(site: PerturbSite, motif_set: MotifSet) -> int:
    pwm = motif_set.pwms.get(site.motif)
    return pwm.length if pwm else site.core_len


def _consensus_bytes(motif: str, motif_set: MotifSet) -> np.ndarray:
    """Consensus of a *real* matrix.  Missing is an error, not a guess.

    The previous version fell back to a hand-typed 'GGTAAGT'.  That is exactly
    the sort of invented sequence this package no longer contains: an inserted
    motif whose letters came from nobody's data would make every sufficiency
    result untraceable.
    """
    pwm = motif_set.pwms.get(motif)
    if pwm is None:
        raise KeyError(
            f"motif {motif!r} is not in the loaded motif set ({sorted(motif_set.pwms)}); "
            "insertion needs a real matrix to take its consensus from"
        )
    return _as_bytes(pwm.consensus())


def _insertion_series(ref: np.ndarray, seed: int, spacings: Sequence[int],
                      copies: Sequence[int], motif_set: MotifSet, do_insert: bool,
                      do_spacing: bool, motif: Optional[str] = None) -> List[PerturbedSeq]:
    """Plant a consensus motif into a dinucleotide-matched neutral background."""
    rng = np.random.default_rng(seed + 101)
    bg = v_shuffle_di(ref, rng)          # composition-matched, structure-free
    if motif is None:
        motif = DONOR_NAME if DONOR_NAME in motif_set.pwms else next(iter(motif_set.pwms))
    cons = _consensus_bytes(motif, motif_set)
    mid = len(bg) // 2
    out: List[PerturbedSeq] = [PerturbedSeq(bg.copy(), dict(
        pert_family="insert_background", pert_site=-1, pert_motif=motif,
        pert_param="n=0", pert_pair_id="ins"))]

    if do_insert:
        for n in copies:
            s = bg.copy()
            for c in range(n):
                at = mid + c * 60
                if at + len(cons) < len(s):
                    s[at : at + len(cons)] = cons
            out.append(PerturbedSeq(s, dict(
                pert_family="insert", pert_site=-1, pert_motif=motif,
                pert_param=f"n={n}", pert_pair_id="ins",
                pert_core_offset=mid)))

    if do_spacing:
        for sp in spacings:
            s = bg.copy()
            for at in (mid, mid + sp + len(cons)):
                if at + len(cons) < len(s):
                    s[at : at + len(cons)] = cons
            out.append(PerturbedSeq(s, dict(
                pert_family="spacing", pert_site=-1, pert_motif=motif,
                pert_param=f"gap={sp}", pert_pair_id="ins",
                pert_core_offset=mid)))
    return out


# --------------------------------------------------------------------------
# site selection
# --------------------------------------------------------------------------

DONOR_NAME = "splice_donor_U2"
ACCEPTOR_NAME = "splice_acceptor_U2"

#: Which matrix applies to a site, chosen by the dinucleotide actually present.
#: A GC-AG acceptor is indistinguishable from a U2 acceptor at the acceptor end
#: alone -- both read AG -- so acceptors map to the U2 matrix unless the core is
#: AC.  Pairing the two ends would resolve it and is not done here.
DONOR_BY_CORE = {"GT": "splice_donor_U2", "GC": "splice_donor_GCAG",
                 "AT": "splice_donor_ATAC"}
ACCEPTOR_BY_CORE = {"AG": "splice_acceptor_U2", "AC": "splice_acceptor_ATAC"}


def sites_from_context(
    context: np.ndarray,
    scored_start: int,
    window_start: int,
    motif_set: MotifSet,
    cores: Optional[Sequence[str]] = None,
    max_sites: int = 4,
    flank: int = 100,
    window_bp: int = 6000,
) -> List[PerturbSite]:
    """Pick annotated splice sites inside a window as perturbation targets.

    The motif geometry (where the core sits inside the matrix) comes from the
    loaded matrix, so a set built with different window conventions stays
    self-consistent instead of inheriting offsets from somewhere else.
    """
    motif_set.require("perturbation site selection")
    skipped_no_model = 0
    sites: List[PerturbSite] = []
    for i, ctx in enumerate(context):
        if ctx not in ("splice_donor", "splice_acceptor"):
            continue
        core_global = scored_start + i
        offset = core_global - window_start
        if offset - flank < 0 or offset + flank >= window_bp:
            continue
        core = (cores[i] if cores is not None and i < len(cores) else "") or ""
        table = DONOR_BY_CORE if ctx == "splice_donor" else ACCEPTOR_BY_CORE
        motif = table.get(core.upper(),
                          DONOR_NAME if ctx == "splice_donor" else ACCEPTOR_NAME)
        pwm = motif_set.pwms.get(motif)
        if pwm is None:
            # The site's own class has no matrix in this set.  Skip it and count
            # it rather than perturbing it under a matrix for a different class.
            skipped_no_model += 1
            continue
        sites.append(PerturbSite(
            offset=offset - pwm.core_offset,
            motif=motif,
            core_offset=pwm.core_offset,
            core_len=pwm.core_len,
            flank=flank,
            label=ctx,
        ))
        if len(sites) >= max_sites:
            break
    if skipped_no_model:
        sites_skipped_counter[0] += skipped_no_model
    return sites


#: Sites passed over because no matrix covers their class.  Read and reported by
#: the extraction so the count appears in the run log rather than nowhere.
sites_skipped_counter = [0]


def paired_effect_frame(df, value: str, family: str = "core_mut",
                        reference: str = "reference"):
    """Paired difference between a perturbation and its own reference sequence.

    The pairing is on ``window_id`` x ``pert_pair_id`` x ``offset``, so the two
    sequences differ only by the edit and every other covariate is held fixed by
    construction.
    """
    import pandas as pd

    key = ["window_id", "pert_pair_id", "offset"]
    ref = df[df["pert_family"] == reference][key + [value]].rename(columns={value: "ref"})
    alt = df[df["pert_family"] == family][key + [value, "pert_motif", "pert_label", "context"]]
    merged = alt.merge(ref, on=key, how="inner")
    merged["delta"] = merged[value] - merged["ref"]
    return merged
