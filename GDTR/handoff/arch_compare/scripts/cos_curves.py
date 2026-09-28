"""Per-region layer-wise cosine curves under three references, 7B and 40B.

  consecutive : cos(h_l, h_{l-1})       how much the direction turns per block
  to_hnorm    : cos(h_l, h_norm)        the workshop paper's reference
  to_preonset : cos(h_l, h_{onset-1})   last block before the norm cascade
7B: layers 0-23 from the fp16 probe cache (finite there), 24-31 + h_norm from the
fp32 re-extraction (identical rows, verified earlier). 40B: all fp32.
"""
import numpy as np, zipfile, struct, json
R = "/path/to/TDiG/arch_compare/results"
CODES = {0: "intergenic", 1: "intron", 2: "coding_exon", 5: "splice_donor", 6: "splice_acceptor"}
CAP = 3000

def open_H(path):
    z = zipfile.ZipFile(path); zi = z.getinfo("H.npy")
    with z.open("H.npy") as fh:
        ver = np.lib.format.read_magic(fh); shp, fo, dt = np.lib.format._read_array_header(fh, ver); hdr = fh.tell()
    z.fp.seek(zi.header_offset); n, m = struct.unpack("<HH", z.fp.read(30)[26:30])
    return np.memmap(path, dtype=dt, mode="r", offset=zi.header_offset + 30 + n + m + hdr, shape=shp)

def rows_for(lab, seed):
    rng = np.random.default_rng(seed); out = {}
    for c, nm in CODES.items():
        jj = np.where(lab == c)[0]
        if len(jj) == 0: continue
        out[nm] = np.sort(rng.choice(jj, min(CAP, len(jj)), replace=False))
    return out

def cosrow(a, b):
    na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / np.clip(na * nb, 1e-30, None)

def summarize(vals):
    v = vals[np.isfinite(vals)]
    return [float(np.mean(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))] if len(v) else [None] * 3

out = {}
for ch, pfx in (("chr22", "evo2_7b"), ("chr17", "chr17_evo2_7b")):
    late = np.load(f"{R}/{ch}_late_fp32.npz"); lab = late["label"]
    H16 = open_H(f"{R}/{pfx}_probe_subsample.npz")
    sel = rows_for(lab, 11)
    get = lambda l, r: (np.asarray(H16[l][r], dtype=np.float64) if l < 24 else late[f"h{l}"][r].astype(np.float64))
    res = {}
    for nm, r in sel.items():
        hn = late["hnorm"][r].astype(np.float64); pre = late["h27"][r].astype(np.float64)
        prev = None; cur = {"consecutive": [], "to_hnorm": [], "to_preonset": []}
        for l in range(32):
            h = get(l, r)
            cur["consecutive"].append(summarize(cosrow(h, prev)) if prev is not None else [None] * 3)
            cur["to_hnorm"].append(summarize(cosrow(h, hn)))
            cur["to_preonset"].append(summarize(cosrow(h, pre)))
            prev = h
        res[nm] = cur
    out[f"7B_{ch}"] = {"onset": 28, "curves": res}
    print("done 7B", ch, flush=True)

for ch in ("chr22", "chr17"):
    D = f"{R}/{ch}_evo2_40b"; meta = np.load(f"{D}/meta.npz"); lab = meta["label"]
    prof = json.load(open(f"{D}/profile.json")); NB = prof["blocks"]
    on = next(i + 1 for i, x in enumerate(prof["ratio"]) if x >= 10)
    M = {f"b{i}": np.load(f"{D}/b{i}.npy", mmap_mode="r") for i in range(NB)}; M["norm"] = np.load(f"{D}/norm.npy", mmap_mode="r")
    sel = rows_for(lab, 11); res = {}
    for nm, r in sel.items():
        hn = M["norm"][r].astype(np.float64); pre = M[f"b{on-1}"][r].astype(np.float64)
        prev = None; cur = {"consecutive": [], "to_hnorm": [], "to_preonset": []}
        for l in range(NB):
            h = M[f"b{l}"][r].astype(np.float64)
            cur["consecutive"].append(summarize(cosrow(h, prev)) if prev is not None else [None] * 3)
            cur["to_hnorm"].append(summarize(cosrow(h, hn)))
            cur["to_preonset"].append(summarize(cosrow(h, pre)))
            prev = h
        res[nm] = cur
    out[f"40B_{ch}"] = {"onset": on, "curves": res}
    print("done 40B", ch, flush=True)

json.dump(out, open(f"{R}/cos_curves.json", "w"))
for key, v in out.items():
    print(f"\n{key} (onset {v['onset']}) mean cos to h_norm by region, every 4th layer:")
    for nm, c in v["curves"].items():
        print(f"  {nm:<16}" + " ".join(f"{i}:{x[0]:.2f}" for i, x in enumerate(c['to_hnorm']) if i % 4 == 0 or i == len(c['to_hnorm']) - 1))
