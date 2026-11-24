"""Test: does multi-frame face fusion recognize far-away (small) faces better, without more wrong names?

Like evaluate_cctv.py, every enrolled photo is shrunk so the face is 16 .. 32 px wide, but now it is seen
over FRAMES consecutive frames (a person standing or walking in front of a CCTV camera): each frame gets
a little random movement, scale change, blur, sensor noise and JPEG compression. Each frame goes through
the live CCTV path: face detector on the person's area (640 px), ArcFace, matching.

Two recognizers are compared on exactly the same frames:
  old: one frame at a time; a face must be >= 24 px and score >= MATCH_THRESHOLD by itself
  new: the same, plus the weighted average of the person's frames so far (fusion.py, faces from 16 px)
Enrolled students: the photo is matched against all OTHER photos (never itself or a duplicate of it).
Strangers: the same sequences, with that student removed from the gallery: any name given is a false accept.

Usage: python evaluate_fusion.py            (results also in data/fusion_results.csv)
"""
import csv

import cv2
import numpy as np

import config
import fusion
from detection import find_face
from enroll import PHOTO_EXT
from faces import detect_faces, load_face_app

FACE_WIDTHS = [16, 20, 24, 28, 32]
FRAMES = 15
FRAME_W, FRAME_H = 1280, 720


def frames_of(img, box, width, rng):
    """FRAMES small views of one photo (face about `width` px) + the person area around the face."""
    x1, y1, x2, y2 = box
    out = []
    for _ in range(FRAMES):
        s = width / (x2 - x1) * rng.uniform(0.94, 1.06)
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        if sw > FRAME_W or sh > FRAME_H:
            cx, cy = int((x1 + x2) / 2 * s), int((y1 + y2) / 2 * s)
            ax, ay = max(0, min(sw - FRAME_W, cx - FRAME_W // 2)), max(0, min(sh - FRAME_H, cy - FRAME_H // 2))
            small = small[ay:ay + min(sh, FRAME_H), ax:ax + min(sw, FRAME_W)]
            sh, sw = small.shape[:2]
        frame = np.full((FRAME_H, FRAME_W, 3), 90, np.uint8)
        ox, oy = int(rng.integers(0, FRAME_W - sw + 1)), int(rng.integers(0, FRAME_H - sh + 1))
        frame[oy:oy + sh, ox:ox + sw] = small
        sigma = rng.uniform(0, 0.8)
        if sigma > 0.2:
            frame = cv2.GaussianBlur(frame, (0, 0), sigma)
        frame = np.clip(frame + rng.normal(0, 3, frame.shape), 0, 255).astype(np.uint8)
        frame = cv2.imdecode(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 85))])[1], 1)
        out.append((frame, [ox, oy, ox + sw, oy + sh]))
    return out


def decide(samples, gallery, names, mask, use_fusion):
    """Runs one sequence through the old or new rule. Returns (outcome, frames until named or None)."""
    person = {}
    for k, (emb, px, det) in enumerate(samples):
        if emb is None:
            continue
        sims = np.where(mask, gallery @ emb, -1.0)
        best = int(np.argmax(sims))
        if px >= config.CCTV_MIN_FACE_PX and sims[best] >= config.MATCH_THRESHOLD:
            return names[best], k + 1
        if use_fusion and px >= config.FUSE_MIN_FACE_PX:
            avg, n = fusion.add(person, emb, fusion.weight(det, px), now=k * 0.2)
            if n >= config.FUSE_MIN:
                fs = np.where(mask, gallery @ avg, -1.0)
                fb = int(np.argmax(fs))
                if fs[fb] >= config.MATCH_THRESHOLD:
                    return names[fb], k + 1
    return "unknown", None


def main():
    app = load_face_app()
    photos = []
    for folder in sorted(p for p in config.ENROLLMENT_DIR.iterdir() if p.is_dir()):
        for f in sorted(x for x in folder.iterdir() if x.suffix.lower() in PHOTO_EXT):
            img = cv2.imread(str(f))
            faces = detect_faces(app, img) if img is not None else []
            if faces:
                face = max(faces, key=lambda q: q.bbox[2] - q.bbox[0])
                photos.append((folder.name.lower(), img, face.bbox, face.normed_embedding))
    gallery = np.array([p[3] for p in photos])
    names = [p[0] for p in photos]
    dup = gallery @ gallery.T > 0.97                      # the same shot saved twice counts as itself

    rng = np.random.default_rng(0)
    rows = []
    for i, (who, img, box, _) in enumerate(photos):
        genuine = ~dup[i]                                  # all other photos (not this one or its duplicates)
        stranger = np.array([n != who for n in names])     # this student removed: they are a stranger now
        for width in FACE_WIDTHS:
            samples = []
            for frame, area in frames_of(img, box, width, rng):
                face = find_face(frame, area, app, 640, config.FUSE_MIN_FACE_PX)
                if face is None:
                    samples.append((None, 0, 0))
                    continue
                app.models["recognition"].get(frame, face)
                samples.append((face.normed_embedding, float(face.bbox[2] - face.bbox[0]), float(face.det_score)))
            found = sum(s[0] is not None for s in samples)
            for test, mask, truth in (("enrolled", genuine, who), ("stranger", stranger, None)):
                for rule in ("old", "new"):
                    name, k = decide(samples, gallery, names, mask, rule == "new")
                    outcome = ("correct" if name == truth else "wrong") if name != "unknown" else "unknown"
                    if truth is None:
                        outcome = "false accept" if name != "unknown" else "rejected"
                    rows.append({"student": who, "photo": i, "face_px": width, "test": test, "rule": rule,
                                 "outcome": outcome, "frames": k, "faces_found": found})

    out = config.DATA_DIR / "fusion_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n_seq = len(photos)
    print(f"{n_seq} photos of {len(set(names))} students, {FRAMES} frames each, faces {FACE_WIDTHS} px\n")
    print("ENROLLED students (should be named):  correct / unknown / WRONG   (avg frames until named)")
    for width in FACE_WIDTHS:
        line = f"  {width:>2} px face:"
        for rule in ("old", "new"):
            r = [x for x in rows if x["face_px"] == width and x["test"] == "enrolled" and x["rule"] == rule]
            c = {o: sum(x["outcome"] == o for x in r) for o in ("correct", "unknown", "wrong")}
            ks = [x["frames"] for x in r if x["outcome"] == "correct"]
            avg = f"{np.mean(ks):.1f}" if ks else "-"
            line += f"   {rule}: {c['correct']:>2}/{c['unknown']:>2}/{c['wrong']} ({100 * c['correct'] // len(r):>3}%, {avg} fr)"
        print(line)
    print("\nSTRANGERS (student removed from the gallery; should stay unknown):  false accepts")
    for width in FACE_WIDTHS:
        line = f"  {width:>2} px face:"
        for rule in ("old", "new"):
            r = [x for x in rows if x["face_px"] == width and x["test"] == "stranger" and x["rule"] == rule]
            line += f"   {rule}: {sum(x['outcome'] == 'false accept' for x in r)} of {len(r)}"
        print(line)
    print("\nSaved", out)


if __name__ == "__main__":
    main()
