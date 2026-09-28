"""Full-pipeline rehearsal on a synthetic genome and a rehearsal model.

What this is for
----------------
Before burning GPU hours, run *every step in order* on data small enough to
finish in a minute, with a model that has the same module structure and the same
planted phenomenon.  The unit tests check each function; this checks that the
functions agree with each other -- tap keys, column names, panel geometry,
degenerate groups, file formats, the order of the steps.  Those are the failures
that cost a night on a cluster, and none of them is visible to a unit test.

It is a rehearsal, not a simulation.  The synthetic genome carries real splice
grammar (three intron classes, start codons, polyadenylation hexamers) so the
motif acquisition has something genuine to count and its consensus assertions
can fire; the rehearsal model carries a planted handoff so the analyses have
something to find.  Neither is Evo 2, and no number produced here is evidence
about Evo 2.  The output is a pass/fail list of steps, not a result.

Running it
----------
``python -m exp1 dryrun``  -- everything, in a temporary directory.
``python -m exp1 dryrun --keep /tmp/rehearsal``  -- leave the tree behind to
inspect the tables and figures each step wrote.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

RC = str.maketrans("ACGT", "TGCA")


def _rc(t: str) -> str:
    return t[::-1].translate(RC)


# --------------------------------------------------------------------------
# synthetic reference
# --------------------------------------------------------------------------

DONOR_P = np.array([
    [.35, .35, .18, .12], [.60, .13, .14, .13], [.09, .03, .80, .08],
    [0, 0, 1, 0], [0, 0, 0, 1],
    [.53, .03, .42, .02], [.71, .08, .12, .09], [.07, .06, .81, .06],
    [.16, .16, .19, .49]])

KOZAK = "GCCGCC" + "ATG" + "G"
PAS = "GCAATAAATTGC"


def _draw(P: np.ndarray, rng: np.random.Generator) -> str:
    return "".join("ACGT"[rng.choice(4, p=r / r.sum())] for r in P)


def _donor(core: str, rng: np.random.Generator) -> str:
    P = DONOR_P.copy()
    P[3] = 0.0
    P[4] = 0.0
    P[3, "ACGT".index(core[0])] = 1.0
    P[4, "ACGT".index(core[1])] = 1.0
    return _draw(P, rng)


def _acceptor(core: str, rng: np.random.Generator) -> str:
    m = np.vstack([np.tile([.10, .32, .09, .49], (18, 1)),
                   np.zeros((2, 4)), np.tile([.25] * 4, (3, 1))])
    m[18, "ACGT".index(core[0])] = 1.0
    m[19, "ACGT".index(core[1])] = 1.0
    return _draw(m, rng)


def write_reference(out: Path, motif_units: int = 1100, panel_kb: int = 260,
                    spacing: int = 400, seed: int = 7) -> Tuple[Path, Path, Dict]:
    """Two chromosomes: one to COUNT motifs on, one to run the panel on.

    They are separate on purpose.  The acquisition step refuses to derive a
    motif model on a panel chromosome, and a rehearsal that quietly used one
    chromosome for both would exercise the wrong path.
    """
    rng = np.random.default_rng(seed)
    plan = [("GT", "AG", int(motif_units * 0.70)),
            ("GC", "AG", int(motif_units * 0.24)),
            ("AT", "AC", max(int(motif_units * 0.06), 60))]
    units = [(d, a) for d, a, n in plan for _ in range(n)]
    rng.shuffle(units)

    def genome(n_units: int, name: str, with_features: bool) -> Tuple[str, List[str]]:
        L = (n_units + 2) * spacing
        g = list("".join(rng.choice(list("ACGT"), size=L, p=[.29, .21, .21, .29])))
        gtf: List[str] = []

        def put(at: int, t: str) -> None:
            g[at:at + len(t)] = list(t)

        for i in range(n_units):
            b = i * spacing
            e1, s2 = b + 100, b + 240
            strand = "+" if i % 2 == 0 else "-"
            dcore, acore = units[i % len(units)]
            d, a = _donor(dcore, rng), _acceptor(acore, rng)
            tstart, tend = b + 1, b + 340
            if strand == "+":
                put(e1 - 3, d)
                put(s2 - 20, a)
                put(b + 250, KOZAK)
                sc_lo, sc_hi = b + 257, b + 259
                put(b + 300, PAS)
            else:
                put(s2 - 6, _rc(d))
                put(e1 - 3, _rc(a))
                put(b + 250, _rc(KOZAK))
                sc_lo, sc_hi = b + 252, b + 254
                put(b + 18, _rc(PAS))
            tid = f"{name}_tx{i}"
            gtf += [
                f'{name}\tr\tgene\t{tstart}\t{tend}\t.\t{strand}\t.\tgene_id "{name}_g{i}";',
                f'{name}\tr\ttranscript\t{tstart}\t{tend}\t.\t{strand}\t.\tgene_id "{name}_g{i}"; transcript_id "{tid}";',
                f'{name}\tr\texon\t{tstart}\t{e1}\t.\t{strand}\t.\tgene_id "{name}_g{i}"; transcript_id "{tid}";',
                f'{name}\tr\texon\t{s2 + 1}\t{tend}\t.\t{strand}\t.\tgene_id "{name}_g{i}"; transcript_id "{tid}";',
                f'{name}\tr\tCDS\t{b + 20}\t{e1}\t.\t{strand}\t.\tgene_id "{name}_g{i}"; transcript_id "{tid}";',
                f'{name}\tr\tstart_codon\t{sc_lo}\t{sc_hi}\t.\t{strand}\t.\tgene_id "{name}_g{i}"; transcript_id "{tid}";',
            ]
            if with_features:
                gtf.append(
                    f'{name}\tr\tthree_prime_utr\t{e1 + 1}\t{e1 + 30}\t.\t{strand}\t.\t'
                    f'gene_id "{name}_g{i}"; transcript_id "{tid}";')
        return "".join(g), gtf

    seq_m, gtf_m = genome(motif_units, "chrM", with_features=False)
    seq_p, gtf_p = genome(max(panel_kb * 1000 // spacing, 40), "chrS", with_features=True)

    out.mkdir(parents=True, exist_ok=True)
    fasta = out / "rehearsal.fa"
    with fasta.open("w") as fh:
        for name, seq in (("chrM", seq_m), ("chrS", seq_p)):
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")
    gtf = out / "rehearsal.gtf"
    gtf.write_text("\n".join(gtf_m + gtf_p) + "\n")
    return fasta, gtf, dict(chrM_bp=len(seq_m), chrS_bp=len(seq_p),
                            units=motif_units, classes=dict(
                                (f"{d}-{a}", n) for d, a, n in plan))


def write_configs(root: Path, fasta: Path, gtf: Path, n_windows: int,
                  window_bp: int, scored_bp: int) -> None:
    cfg = root / "configs"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "models.yaml").write_text(
        "models:\n"
        "  mock:\n"
        "    hf_name: mock\n"
        "    n_blocks: 32\n"
        "    width: 64\n"
        "    dtype: float32\n"
        "    block_module_path: model.blocks\n"
        "    norm_module_path: model.norm\n"
        "    unembed_module_path: model.unembed\n"
        "    mixer_attr: mixer\n"
        "    mlp_attr: mlp\n"
        "    notes: rehearsal model, not Evo 2\n")
    (cfg / "panel.yaml").write_text(
        "panel:\n"
        "  chroms: [chrS]\n"
        f"  n_windows: {n_windows}\n"
        f"  window_bp: {window_bp}\n"
        f"  scored_bp: {scored_bp}\n"
        "  seed: 42\n"
        "  max_n_fraction: 0.02\n"
        f"  fasta: {fasta}\n"
        f"  gtf: {gtf}\n"
        "  phylop_bw: ''\n"
        "  legacy_windows: ''\n")


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

@dataclass
class StepResult:
    name: str
    argv: List[str]
    ok: bool
    seconds: float
    tail: str
    critical: bool = True


def steps(model: str, chrom: str, onset: int, rotation: int,
          motif_chroms: str, n_windows: int = 10) -> List[Tuple[str, List[str], bool]]:
    """Every step, in the order the real run uses.  (name, argv, critical)."""
    m = ["--model", model, "--chrom", chrom, "--device", "cpu"]
    return [
        ("selftest", ["selftest"], True),
        ("step0-motifs plan", ["step0-motifs", "plan"], True),
        ("step0-motifs acquire", ["step0-motifs", "acquire", "--chroms", motif_chroms,
                                  "--no-network"], True),
        ("step0-motifs verify", ["step0-motifs", "verify"], True),
        ("step1-blockmap", ["step1-blockmap"], False),
        ("step4-panel", ["step4-panel", "--chrom", chrom,
                         "--n-windows", str(n_windows)], True),
        ("step2-onset", ["step2-onset", *m, "--n-windows", "3"], True),
        ("step3-fidelity", ["step3-fidelity", *m, "--n-windows", "2"], True),
        ("step5-uref", ["step5-uref", *m, "--n-windows", "4"], True),
        ("step6-prereg", ["step6-prereg", "--force"], True),
        ("step5-extract", ["step5-extract", *m, "--variants", "real,shuffle_di",
                           "--raw-per-context", "60"], True),
        ("step5-extract perturb", ["step5-extract", *m, "--variants", "real",
                                   "--perturb", "--perturb-every", "2",
                                   "--perturb-sites", "1", "--tag", "perturb",
                                   "--raw-per-context", "0"], True),
        ("step5-extract shuffled", ["step5-extract", *m, "--variants", "real",
                                    "--shuffle-seed", "1234", "--tag", "shuffled",
                                    "--raw-per-context", "0"], False),
        ("step7-analysis", ["step7-analysis", *m, "--allow-unfrozen"], True),
        ("step7b-regimes", ["step7b-regimes", *m], True),
        ("step7c-motifs", ["step7c-motifs", *m, "--tag", "perturb"], True),
        ("step7d-dualref", ["step7d-dualref", *m], True),
        ("step7e-sitedefs", ["step7e-sitedefs", *m], True),
        ("step8-stage2", ["step8-stage2", *m, "--allow-unfrozen"], True),
        ("step9-profile", ["step9-profile", *m, "--motif-probes"], True),
    ]


def run(root: Path, n_windows: int = 10, window_bp: int = 2000,
        scored_bp: int = 1000, motif_units: int = 1100,
        stop_on_fail: bool = False, verbose: bool = False) -> Dict[str, object]:
    """Build the rehearsal tree and run every step, reporting pass/fail."""
    root.mkdir(parents=True, exist_ok=True)
    ref = root / "reference"
    fasta, gtf, stats = write_reference(ref, motif_units=motif_units)
    write_configs(root, fasta, gtf, n_windows, window_bp, scored_bp)

    env = dict(os.environ)
    env.update(EXP1_ROOT=str(root), EXP1_RESULTS=str(root / "results"),
               EXP1_CACHE=str(root / "cache"),
               PYTHONPATH=str(Path(__file__).resolve().parents[1]))

    results: List[StepResult] = []
    for name, argv, critical in steps("mock", "chrS", 28, 30, "chrM", n_windows):
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-m", "exp1", *argv], env=env,
                              capture_output=True, text=True, cwd=str(root))
        dt = time.time() - t0
        out = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0
        tail = "\n".join(out.strip().splitlines()[-18:])
        results.append(StepResult(name, argv, ok, dt, tail, critical))
        mark = "PASS" if ok else ("FAIL" if critical else "warn")
        print(f"  [{mark}] {name:26s} {dt:6.1f}s")
        if verbose or not ok:
            for line in tail.splitlines():
                print(f"        {line}")
        if not ok and critical and stop_on_fail:
            break

    n_fail = sum(1 for r in results if not r.ok and r.critical)
    return dict(root=str(root), reference=stats,
                n_steps=len(results), n_failed_critical=n_fail,
                steps=[dict(name=r.name, ok=r.ok, critical=r.critical,
                            seconds=round(r.seconds, 2),
                            argv=" ".join(r.argv),
                            tail=(r.tail if not r.ok else ""))
                       for r in results])


def main(keep: Optional[str] = None, **kw) -> int:
    root = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="exp1-rehearsal-"))
    print(f"rehearsal in {root}\n")
    try:
        rep = run(root, **kw)
    finally:
        pass
    print(f"\n{rep['n_steps'] - rep['n_failed_critical']}/{rep['n_steps']} steps ok, "
          f"{rep['n_failed_critical']} critical failures")
    (root / "rehearsal_report.json").write_text(json.dumps(rep, indent=2))
    print(f"report: {root / 'rehearsal_report.json'}")
    if not keep:
        print("(pass --keep <dir> to leave the tree behind)")
    return 1 if rep["n_failed_critical"] else 0
