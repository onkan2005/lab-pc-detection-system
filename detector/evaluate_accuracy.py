"""How accurate is face recognition, and is MATCH_THRESHOLD set right? (numbers for the report)

Uses the enrolled photos themselves, so nothing new has to be collected:

  - Enrolled student (genuine) test, leave-one-out: each photo is taken out and recognized against all the
    OTHER photos. Right if it gets its own name. Photos almost identical to the probe (similarity > 0.97,
    the same shot saved twice) are also taken out, so a duplicate cannot make the result look better.
  - Stranger (impostor) test, leave-one-PERSON-out: all of one student's photos are taken out of the
    gallery, so that student is now a stranger to the system. Right if the face is rejected ("unknown");
    wrong if it is given someone else's name (a false accept, the worst mistake).
  - Optional extra strangers: a folder of face photos of people who are NOT enrolled (--strangers DIR);
    each is recognized against the whole gallery.

For every threshold from 0.20 to 0.70 it reports: recall (enrolled faces correctly named), false accept rate
(strangers given a name), wrong-name rate, precision (of all names given, how many were right) and accuracy,
plus the confusion matrix at the current threshold and the score distributions.

Usage:
  python evaluate_accuracy.py                       # enrolled photos only
  python evaluate_accuracy.py --strangers ../test_faces/strangers
"""
import argparse
import csv
import json

import numpy as np

import config

DUPLICATE_SIM = 0.97
THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.701, 0.05)]


def _best(sims, names, mask):
    """Best match per enrolled person among the gallery photos in `mask`: (name, score, second score)."""
    per = {}
    for s, n, m in zip(sims, names, mask):
        if m and s > per.get(n, -1.0):
            per[n] = float(s)
    ranked = sorted(per.items(), key=lambda kv: -kv[1])
    if not ranked:
        return None, -1.0, None
    return ranked[0][0], ranked[0][1], (ranked[1][1] if len(ranked) > 1 else None)


def trials(emb, names, strangers=None):
    """Every recognition attempt: {"true": name or None (stranger), "best", "score", "kind"}."""
    names = [str(n) for n in names]
    emb = np.asarray(emb, dtype=np.float32)
    sims = emb @ emb.T
    out, dropped = [], 0
    people = sorted(set(names))
    for i, who in enumerate(names):
        # genuine: everyone else's photos + this person's OTHER photos (no duplicates of this shot)
        mask = np.ones(len(names), bool)
        mask[i] = False
        dup = sims[i] > DUPLICATE_SIM
        dup[i] = False
        dropped += int(dup.sum())
        mask &= ~dup
        if any(m and n == who for m, n in zip(mask, names)):   # needs at least one other photo of them
            best, score, second = _best(sims[i], names, mask)
            out.append({"true": who, "best": best, "score": score, "second": second, "kind": "enrolled"})
        # impostor: this person removed from the gallery entirely
        if len(people) > 1:
            mask = np.array([n != who for n in names])
            best, score, second = _best(sims[i], names, mask)
            out.append({"true": None, "best": best, "score": score, "second": second, "kind": "left-out student",
                        "who": who})
    for k, e in enumerate(strangers if strangers is not None else []):
        best, score, second = _best(emb @ e, names, np.ones(len(names), bool))
        out.append({"true": None, "best": best, "score": score, "second": second, "kind": "stranger", "who": f"stranger {k + 1}"})
    return out, dropped


def judge(t, th):
    """right / unknown (enrolled but rejected) / wrong (enrolled, someone else's name) /
    rejected (stranger correctly rejected) / false_accept (stranger given a name)."""
    named = t["score"] >= th
    if t["true"] is None:
        return "false_accept" if named else "rejected"
    if not named:
        return "unknown"
    return "right" if t["best"] == t["true"] else "wrong"


def rates(ts, th):
    c = {k: 0 for k in ("right", "unknown", "wrong", "rejected", "false_accept")}
    for t in ts:
        c[judge(t, th)] += 1
    enrolled = c["right"] + c["unknown"] + c["wrong"]
    strangers = c["rejected"] + c["false_accept"]
    given = c["right"] + c["wrong"] + c["false_accept"]
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    return {"threshold": th, **c,
            "recall": pct(c["right"], enrolled),                         # enrolled faces correctly named
            "far": pct(c["false_accept"], strangers),                   # strangers given a name
            "wrong_rate": pct(c["wrong"], enrolled),                    # enrolled faces given ANOTHER student's name
            "precision": pct(c["right"], given),                        # names given that were right
            "accuracy": pct(c["right"] + c["rejected"], enrolled + strangers)}


def measure(emb, names, strangers=None, threshold=config.MATCH_THRESHOLD):
    ts, dropped = trials(emb, names, strangers)
    table = [rates(ts, th) for th in THRESHOLDS]
    current = rates(ts, threshold)
    people = sorted(set(str(n) for n in names))
    labels = people + ["unknown"]
    matrix = {who: {l: 0 for l in labels} for who in people + ["not enrolled"]}
    for t in ts:
        row = t["true"] or "not enrolled"
        col = t["best"] if t["score"] >= threshold else "unknown"
        matrix[row][col] += 1
    gen = sorted(round(t["score"], 3) for t in ts if t["true"] is not None and t["best"] == t["true"])
    imp = sorted(round(t["score"], 3) for t in ts if t["true"] is None)
    safe = [r for r in table if r["false_accept"] == 0 and r["wrong"] == 0]
    return {
        "threshold": threshold, "current": current, "table": table, "matrix": matrix, "labels": labels,
        "people": people, "photos": len(names), "duplicates_removed": dropped // 2,
        "trials": {"enrolled": sum(t["true"] is not None for t in ts),
                   "left_out": sum(t["kind"] == "left-out student" for t in ts),
                   "strangers": sum(t["kind"] == "stranger" for t in ts)},
        "scores": {"genuine": gen, "impostor": imp,
                   "genuine_min": gen[0] if gen else None, "impostor_max": imp[-1] if imp else None},
        "safe_range": [safe[0]["threshold"], safe[-1]["threshold"]] if safe else None,
        # how far the threshold is above the most similar stranger: the safety margin against false accepts
        "margin": round(threshold - imp[-1], 3) if imp else None,
        "hardest": sorted(({"who": t.get("who") or t["true"], "kind": t["kind"], "best": t["best"], "score": round(t["score"], 3)}
                           for t in ts if t["true"] is None), key=lambda x: -x["score"])[:5],
    }


def stranger_embeddings(folder):
    """ArcFace embeddings of every face photo in a folder (one face per photo: the largest)."""
    import cv2
    from enroll import PHOTO_EXT
    from faces import detect_faces, load_face_app
    app = load_face_app()
    out = []
    for f in sorted(p for p in folder.rglob("*") if p.suffix.lower() in PHOTO_EXT):
        img = cv2.imread(str(f))
        faces = detect_faces(app, img) if img is not None else []
        if faces:
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            out.append(face.normed_embedding)
    return np.array(out)


def main():
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strangers", type=Path, help="folder of face photos of people who are NOT enrolled")
    ap.add_argument("--out", type=Path, default=config.DATA_DIR / "accuracy")
    args = ap.parse_args()
    data = np.load(config.EMBEDDINGS_FILE)
    strangers = stranger_embeddings(args.strangers) if args.strangers else None
    r = measure(data["embeddings"], data["names"], strangers)

    tr = r["trials"]
    print(f"Enrolled: {len(r['people'])} people, {r['photos']} photos ({r['duplicates_removed']} duplicate pairs not used)")
    print(f"Tests: {tr['enrolled']} enrolled-student, {tr['left_out']} left-out-student (stranger), {tr['strangers']} extra stranger")
    print(f"\n{'thresh':>6} {'recall':>7} {'FAR':>6} {'wrong':>6} {'precis':>7} {'accur':>6}")
    for row in r["table"]:
        mark = "  <- current" if abs(row["threshold"] - r["threshold"]) < 1e-6 else ""
        print(f"{row['threshold']:>6.2f} {row['recall']!s:>6}% {row['far']!s:>5}% {row['wrong_rate']!s:>5}% "
              f"{row['precision']!s:>6}% {row['accuracy']!s:>5}%{mark}")
    print(f"\nConfusion matrix at threshold {r['threshold']} (rows: who it really is, columns: what the system said)")
    print(" " * 14 + "".join(f"{l[:10]:>11}" for l in r["labels"]))
    for row, cols in r["matrix"].items():
        print(f"{row[:13]:<14}" + "".join(f"{cols[l]:>11}" for l in r["labels"]))
    s = r["scores"]
    print(f"\nLowest same-person score {s['genuine_min']}, highest stranger score {s['impostor_max']}")
    if r["safe_range"]:
        print(f"Thresholds with no false accepts and no wrong names here: {r['safe_range'][0]}..{r['safe_range'][1]}")
    if r["margin"] is not None:
        print(f"Safety margin: the threshold {r['threshold']} is {r['margin']} above the most similar stranger "
              f"(with only {tr['left_out'] + tr['strangers']} stranger tests, keep a wide margin; do not lower it to the edge)")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "thresholds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r["table"][0]))
        w.writeheader()
        w.writerows(r["table"])
    (args.out / "accuracy.json").write_text(json.dumps(r, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    bins = np.linspace(-0.2, 1.0, 49)
    ax[0].hist(s["impostor"], bins, alpha=0.7, label="strangers (should be rejected)", color="#e03131")
    ax[0].hist(s["genuine"], bins, alpha=0.7, label="enrolled students (should be named)", color="#1f9d55")
    ax[0].axvline(r["threshold"], color="k", ls="--", label=f"threshold {r['threshold']}")
    ax[0].set_xlabel("ArcFace cosine similarity of the best match"); ax[0].set_ylabel("faces"); ax[0].legend(fontsize=8)
    th = [x["threshold"] for x in r["table"]]
    for key, label, color in (("recall", "recall", "#1f9d55"), ("far", "false accept rate", "#e03131"),
                              ("precision", "precision", "#3b82f6")):
        ax[1].plot(th, [x[key] for x in r["table"]], marker="o", ms=3, label=label, color=color)
    ax[1].axvline(r["threshold"], color="k", ls="--"); ax[1].set_xlabel("threshold"); ax[1].set_ylabel("%")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "accuracy.png", dpi=110)
    print(f"\nSaved {args.out / 'thresholds.csv'}, accuracy.json and accuracy.png")


if __name__ == "__main__":
    main()
