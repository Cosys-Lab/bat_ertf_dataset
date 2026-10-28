"""Out-of-sample check of the placement rule (referee item M2), from the stored placement-study data (no BEM runs).

The published rule (docs/placement_rule.md, scripts/apply_placement_rule.py) has no continuous fitted parameters:
what was "learned" on the four calibration ears and two emitters are discrete choices (which descriptors, equal
weights, the smoothing radius; for the nostrils the score, the spacing window and the mirror tolerance). To test how
well such a choice transfers, we re-make it on a training subset and apply it to the held-out data:

- ears: rule family = every single descriptor and every equal-weight sum of two standardized descriptors (front/back
  ratio, directivity index, frontal gain, peak gain, occlusion), each smoothed over 0, 0.5, 0.75 or 1.0 mm
  (60 rules, the published one among them). The rule with the highest mean reference match over the training ears
  is selected (all tied rules are reported) and applied to the held-out ear(s).
- nostrils: rule family = score (front/back ratio, directivity index or frontal gain of the summed field, or summed
  occlusion) x spacing window (1.0-3.5 or 1.5-2.5 mm) x mirror tolerance (0.6 or 1.0 mm); 16 rules. Train on one
  species' emitter, test on the other.
Splits: leave-one-species-out (train on 2 ears / 1 emitter, test on 2 ears / 1 emitter) and leave-one-ear-out
(train on 3 ears; note that the held-out ear's mirror partner stays in training).

Reference match: pattern correlation with the earlier reference simulation in the refit frame of placement_study.py
(frontal hemisphere, dB re maximum clipped at -30 dB, mean over the ten study frequencies). For each held-out ear or
emitter: the selected candidate's match, the best candidate, the median candidate, the most occluded candidate, and
the selected candidate's percentile among all candidates (= chance that a random candidate is not better).
Nostril reference set: all candidate pairs 0.8-4 mm apart (as in placement_study.py); also the percentile among the
mirror-symmetric pairs admissible to the published rule.
Usage: python scripts/placement_out_of_sample.py  -> results/validation/placement_oos.json"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batertf import mesh, paths, reference, rotate, symmetry
from learn_placement_rule import descriptors
from test_nostril_rule import mirror

KEYS = ["pdisc", "mmicr"]
EAR_DESC = ["front_back_db", "directivity_db", "gain_front_db", "gain_peak_db", "occlusion"]
SMOOTHS = [0.0, 0.5e-3, 0.75e-3, 1.0e-3]
NOSE_SCORES = ["front_back_db", "directivity_db", "gain_front_db", "occlusion"]
WINDOWS = [(1.0e-3, 3.5e-3), (1.5e-3, 2.5e-3)]
TOLS = [0.6e-3, 1.0e-3]
PUBLISHED_EAR = (("front_back_db", "occlusion"), 0.75e-3)
PUBLISHED_NOSE = ("front_back_db", (1.5e-3, 2.5e-3), 1.0e-3)
OUT = paths.RESULTS / "validation" / "placement_oos.json"
z = lambda v: (v - v.mean()) / v.std()


def corr_mean(sim, ref):
    return float(reference.metrics(sim, ref)["corr"].mean())


def load(key):
    sp = paths.species()[key]
    d = np.load(paths.work_dir() / "placement" / f"{key}.npz")
    names = [str(n) for n in d["names"]]
    H, pos, area, freqs = d["H"], d["pos"], d["area"], d["freqs"]
    st = json.loads((paths.RESULTS / "placement" / f"{key}.json").read_text())
    R = np.array(st.get("start_matrix", np.eye(3))) @ rotate.matrix(*st["refit"]["angles"])
    occ = np.load(paths.work_dir() / "occlusion" / f"{key}.npz")
    ao, tree = occ["ao"], cKDTree(occ["vertices"])
    ears = {}
    for s in ("left", "right"):
        ec = st["refit"]["ear_corr"][s]
        ns = list(ec)
        idx = [names.index(n) for n in ns]
        dsc = [descriptors(H[i]) for i in idx]
        D = {k: np.array([x[k] for x in dsc]) for k in EAR_DESC if k != "occlusion"}
        D["occlusion"] = np.array([ao[tree.query(pos[i])[1]] for i in idx])
        ears[f"{key}_{s}"] = {"names": ns, "P": pos[idx], "match": np.array([ec[n] for n in ns]), "D": D,
                              "previous_rule": ns.index(f"ear{s[0].upper()}000")}
    # nostril pairs: every pair 0.8-4 mm apart, reference match recomputed in the refit frame
    ref = reference.load(sp.get("reference_key", key))
    ki = [int(np.argmin(abs(ref["freqs"] - f))) for f in freqs]
    Rem = ref["emitter"][..., ki]
    m = mesh.load(sp["stl"])
    sym, _ = symmetry.fit(m)
    nose = [i for i, n in enumerate(names) if n.startswith("nose")]
    hn = {i: rotate.hemisphere_from_sphere(H[i].real, R) + 1j * rotate.hemisphere_from_sphere(H[i].imag, R)
          for i in nose}
    pairs = []
    for a, b in itertools.combinations(nose, 2):
        dist = np.linalg.norm(pos[a] - pos[b])
        if 0.8e-3 <= dist <= 4e-3:
            h = H[a] * area[a] + H[b] * area[b]
            dsc = descriptors(h)
            pairs.append({"a": names[a], "b": names[b], "spacing": float(dist),
                          "mismatch": float(np.linalg.norm(mirror(pos[a], *sym) - pos[b])),
                          "match": corr_mean(np.abs(hn[a] * area[a] + hn[b] * area[b]), Rem),
                          "occlusion": float(ao[tree.query(pos[a])[1]] + ao[tree.query(pos[b])[1]]),
                          **{k: float(dsc[k]) for k in NOSE_SCORES if k != "occlusion"}})
    return ears, {"pairs": pairs, "n_nose_candidates": len(nose)}


def ear_score(e, rule):
    descs, smooth = rule
    sc = sum(z(e["D"][k]) for k in descs)
    if smooth > 0:
        t = cKDTree(e["P"])
        sc = np.array([sc[t.query_ball_point(p, smooth)].mean() for p in e["P"]])
    return sc


def ear_rules():
    combos = [(k,) for k in EAR_DESC] + list(itertools.combinations(EAR_DESC, 2))
    return [(c, s) for c in combos for s in SMOOTHS]


def ear_pick(e, rule):
    return int(np.argmax(ear_score(e, rule)))


def nose_pick(pairs, rule):
    score, (lo, hi), tol = rule
    ok = [p for p in pairs if lo <= p["spacing"] <= hi and p["mismatch"] < tol]
    return max(ok, key=lambda p: p[score]) if ok else None


def nose_rules():
    return [(s, w, t) for s in NOSE_SCORES for w in WINDOWS for t in TOLS]


def summary_ear(e, i):
    v = e["match"]
    occ = int(np.argmax(e["D"]["occlusion"]))
    return {"selected": float(v[i]), "selected_name": e["names"][i], "best": float(v.max()),
            "median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "most_occluded": float(v[occ]), "previous_rule_point": float(v[e["previous_rule"]]),
            "percentile": float(100 * np.mean(v <= v[i])),
            "distance_to_best_mm": float(np.linalg.norm(e["P"][i] - e["P"][int(np.argmax(v))]) * 1e3),
            "n_candidates": int(len(v))}


def summary_nose(nd, p):
    allm = np.array([q["match"] for q in nd["pairs"]])
    pub = [q for q in nd["pairs"] if PUBLISHED_NOSE[1][0] <= q["spacing"] <= PUBLISHED_NOSE[1][1]
           and q["mismatch"] < PUBLISHED_NOSE[2]]
    pm = np.array([q["match"] for q in pub])
    occ = max(pub, key=lambda q: q["occlusion"])
    return {"selected": p["match"], "selected_pair": [p["a"], p["b"]], "best": float(allm.max()),
            "median": float(np.median(allm)), "p10": float(np.percentile(allm, 10)),
            "p90": float(np.percentile(allm, 90)), "most_occluded_symmetric_pair": occ["match"],
            "percentile": float(100 * np.mean(allm <= p["match"])), "n_pairs": int(len(allm)),
            "best_symmetric": float(pm.max()), "median_symmetric": float(np.median(pm)),
            "percentile_among_symmetric": float(100 * np.mean(pm <= p["match"])), "n_symmetric": int(len(pm))}


def rule_name(rule):
    if isinstance(rule[0], tuple):
        return " + ".join(f"z({k})" for k in rule[0]) + f", smoothed {rule[1] * 1e3:.2f} mm"
    return f"{rule[0]}, spacing {rule[1][0] * 1e3:.1f}-{rule[1][1] * 1e3:.1f} mm, mirror tol {rule[2] * 1e3:.1f} mm"


def select(rules, train_value):
    vals = np.array([train_value(r) for r in rules])
    best = np.where(vals >= vals.max() - 1e-9)[0]
    return [rules[i] for i in best], float(vals.max())


def main():
    ears, noses = {}, {}
    for key in KEYS:
        e, n = load(key)
        ears.update(e)
        noses[key] = n
        print(key, {k: len(v["names"]) for k, v in e.items()}, "nose pairs", len(n["pairs"]), flush=True)
    er, nr = ear_rules(), nose_rules()
    pick = {(k, r): ear_pick(ears[k], r) for k in ears for r in er}
    res = {"description": __doc__.split("Usage")[0], "ear_rule_family": [rule_name(r) for r in er],
           "nose_rule_family": [rule_name(r) for r in nr]}
    # in-sample: the published rule on every ear / emitter, and the in-sample best rule of the family
    res["published_rule_in_sample"] = {
        "ears": {k: summary_ear(ears[k], pick[(k, PUBLISHED_EAR)]) for k in ears},
        "emitters": {k: summary_nose(noses[k], nose_pick(noses[k]["pairs"], PUBLISHED_NOSE)) for k in KEYS}}
    tied, v = select(er, lambda r: np.mean([ears[k]["match"][pick[(k, r)]] for k in ears]))
    res["family_in_sample"] = {"ears_best_rules": [rule_name(r) for r in tied], "mean_match": v,
                               "published_rule_mean_match": float(np.mean(
                                   [ears[k]["match"][pick[(k, PUBLISHED_EAR)]] for k in ears])),
                               "published_rule_rank": int(1 + sum(np.mean([ears[k]["match"][pick[(k, r)]] for k in ears]) >
                                                                  np.mean([ears[k]["match"][pick[(k, PUBLISHED_EAR)]] for k in ears]) + 1e-9 for r in er))}

    def ear_split(train, test):
        tied, v = select(er, lambda r: np.mean([ears[k]["match"][pick[(k, r)]] for k in train]))
        out = {"train": train, "selected_rules": [rule_name(r) for r in tied], "train_mean_match": v, "test": {}}
        for k in test:
            s = [summary_ear(ears[k], pick[(k, r)]) for r in tied]
            o = dict(s[0])
            if len(tied) > 1:                       # ties: report the range over the tied rules
                o["selected_range_over_tied_rules"] = [min(x["selected"] for x in s), max(x["selected"] for x in s)]
                o["selected"] = float(np.mean([x["selected"] for x in s]))
                o["percentile"] = float(np.mean([x["percentile"] for x in s]))
            out["test"][k] = o
        return out

    res["leave_one_species_out"] = {"ears": {}, "emitters": {}}
    for held in KEYS:
        train = [k for k in ears if not k.startswith(held)]
        res["leave_one_species_out"]["ears"][held] = ear_split(train, [k for k in ears if k.startswith(held)])
        other = [k for k in KEYS if k != held][0]
        tied, v = select(nr, lambda r: nose_pick(noses[other]["pairs"], r)["match"]
                         if nose_pick(noses[other]["pairs"], r) else -1)
        s = [summary_nose(noses[held], nose_pick(noses[held]["pairs"], r)) for r in tied]
        o = dict(s[0])
        if len(tied) > 1:
            o["selected_range_over_tied_rules"] = [min(x["selected"] for x in s), max(x["selected"] for x in s)]
            o["selected"] = float(np.mean([x["selected"] for x in s]))
            o["percentile"] = float(np.mean([x["percentile"] for x in s]))
        res["leave_one_species_out"]["emitters"][held] = {"train": other, "selected_rules": [rule_name(r) for r in tied],
                                                          "train_match": v, "test": o}
    res["leave_one_ear_out"] = {k: ear_split([j for j in ears if j != k], [k]) for k in ears}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))

    f = lambda d: (f"sel {d['selected']:.3f} (pct {d['percentile']:.0f}) best {d['best']:.3f} "
                   f"median {d['median']:.3f} occl {d.get('most_occluded', d.get('most_occluded_symmetric_pair')):.3f}")
    print("published rule, in sample:")
    for k, d in {**res["published_rule_in_sample"]["ears"], **res["published_rule_in_sample"]["emitters"]}.items():
        print(f"  {k:12s}", f(d))
    print("family in sample:", res["family_in_sample"])
    for held, r in res["leave_one_species_out"]["ears"].items():
        print(f"LOSO ears, held out {held}: rules {r['selected_rules']}")
        for k, d in r["test"].items():
            print(f"  {k:12s}", f(d), d.get("selected_range_over_tied_rules", ""))
    for held, r in res["leave_one_species_out"]["emitters"].items():
        print(f"LOSO emitter {held}: rules {r['selected_rules']}\n  ", f(r["test"]),
              f"pct among symmetric {r['test']['percentile_among_symmetric']:.0f}",
              r["test"].get("selected_range_over_tied_rules", ""))
    for k, r in res["leave_one_ear_out"].items():
        print(f"LOEO {k}: rules {r['selected_rules']}\n  ", f(r["test"][k]))
    print(OUT)


if __name__ == "__main__":
    main()
