"""
lora_calibration.py  (v3 - logic only; the Calibration tab lives in the calculator)

A calibration POINT records a dataset the calculator got wrong and the settings that actually worked on it:
    baseline : what the plain calculator recommended for that dataset
    worked   : what worked (rank, alpha, LR before accumulation scaling, best/peak step, timing weight,
               planner scale, caption dropout, decay shape)
    extras   : everything else typed in (steps, decay steps, optimizer, analyser scores, verdicts ...)
One saved point at a time is the ACTIVE calibration "datum".  For a new dataset the calculator keeps its own
curve but is re-anchored so that, on the datum dataset, it reproduces the settings that worked:
        new setting = plain calculation for the new dataset  x  (worked / baseline on the datum dataset)
(dropout is shifted by the same difference instead of scaled).  Everything lives in ONE file, calibration.cfg
(JSON), in the folder the program was started from.
"""
import csv
import json
import os
import shutil
import time

CFG_NAME = "calibration.cfg"
CFG_VERSION = 2
RATIO_KEYS = ("lr", "rank", "steps", "peak", "timing", "planner")
RATIO_MIN, RATIO_MAX = 0.2, 5.0


def cfg_path(folder=None):
    return os.path.join(folder or os.getcwd(), CFG_NAME)


def _empty():
    return {"version": CFG_VERSION, "points": [], "active": None}


def load(path=None):
    path = path or cfg_path()
    if not os.path.exists(path):
        return _empty()
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("points", [])
        data.setdefault("active", None)
        data["version"] = CFG_VERSION
        return data
    except Exception:
        shutil.copy2(path, path + ".corrupt")     # never silently destroy a bad file
        return _empty()


def save(data, path=None):
    path = path or cfg_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def make_point(name, dataset, backend, genre, features, baseline, worked, extras=None):
    return {"id": time.strftime("%Y%m%d-%H%M%S") + "-%03d" % (int(time.time() * 1000) % 1000),
            "name": name, "dataset": dataset, "backend": backend, "genre": (genre or "").strip(),
            "features": dict(features), "baseline": dict(baseline), "worked": dict(worked),
            "extras": dict(extras or {})}


def get_point(data, pid):
    for p in data.get("points", []):
        if p["id"] == pid:
            return p
    return None


def set_active(data, pid):
    data["active"] = pid if get_point(data, pid) else None


def remove_points(data, ids):
    ids = set(ids)
    data["points"] = [p for p in data["points"] if p["id"] not in ids]
    if data.get("active") in ids:
        data["active"] = None


def remove_all(data):
    data["points"] = []
    data["active"] = None


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def anchor_from_point(p):
    """Multipliers/offsets that re-anchor the calculator on this datum point."""
    b, w, ex = dict(p["baseline"]), dict(p["worked"]), p.get("extras", {})
    # points saved by the previous version only stored the peak step; recover steps / decay steps from what was typed
    if not w.get("steps"):
        w["steps"] = _float(ex.get("steps"))
    if not w.get("decay_steps"):
        w["decay_steps"] = _float(ex.get("decay_steps"))
    if not b.get("steps") and b.get("peak"):
        b["steps"] = round(b["peak"] / 0.6 / 10.0) * 10          # plain calculator puts its peak at 60% of the run
    m = {}
    for k in RATIO_KEYS:
        if b.get(k) and w.get(k) and b[k] > 0 and w[k] > 0:
            m[k] = min(RATIO_MAX, max(RATIO_MIN, w[k] / b[k]))
    if "steps" in m:
        m.pop("peak", None)                                       # total steps typed in beats the best-checkpoint estimate
    if w.get("decay_steps") and w.get("steps"):
        m["decay_ratio"] = min(0.8, max(0.05, w["decay_steps"] / w["steps"]))
    if w.get("dropout") is not None and b.get("dropout") is not None:
        m["dropout_delta"] = w["dropout"] - b["dropout"]
    if w.get("alpha") and w.get("rank"):
        m["alpha_ratio"] = w["alpha"] / w["rank"]
    if w.get("decay_shape") in ("Cosine", "Linear"):
        m["decay_shape"] = w["decay_shape"]
    return m


def describe_anchor(m):
    bits = []
    for k, label in (("lr", "LR"), ("rank", "rank"), ("steps", "steps"), ("peak", "peak step"),
                     ("timing", "timing wt"), ("planner", "planner")):
        if k in m:
            bits.append(f"{label} x{m[k]:.2f}")
    if "decay_ratio" in m:
        bits.append(f"decay = {m['decay_ratio']:.0%} of run")
    if "dropout_delta" in m:
        bits.append(f"dropout {m['dropout_delta']:+.2f}")
    if "decay_shape" in m:
        bits.append(f"shape {m['decay_shape']}")
    return ", ".join(bits)


def import_evaluator_csv(path):
    """Read checkpoint_eval_results.csv from the evaluator; return the best-balanced step and its scores."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                vib = r.get("vibe_clap")
                rows.append((int(float(r["step"])), float(r["singer_blend"]),
                             None if vib in (None, "", "None") else float(vib)))
            except (ValueError, KeyError):
                continue
    if not rows:
        raise ValueError("No usable rows in that CSV (expected the evaluator's checkpoint_eval_results.csv).")

    def mm(v):
        lo, hi = min(v), max(v)
        return [(x - lo) / (hi - lo + 1e-9) for x in v] if len(v) > 1 else [1.0] * len(v)

    sing = mm([r[1] for r in rows])
    vib = mm([r[2] for r in rows]) if all(r[2] is not None for r in rows) else sing
    bal = [0.5 * a + 0.5 * b for a, b in zip(sing, vib)]
    i = max(range(len(rows)), key=lambda k: bal[k])
    return {"total_steps": max(r[0] for r in rows), "best_step": rows[i][0],
            "singer": round(rows[i][1], 3), "vibe": None if rows[i][2] is None else round(rows[i][2], 3)}
