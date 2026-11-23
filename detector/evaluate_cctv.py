"""Test: can a CCTV-style camera recognize people who are not looking straight at it, and how far away?

Each enrolled photo is labeled with its head pose (front / left / right / up / down, from its file
name). It is shrunk so the face is 160, 80, 48, 32 or 24 pixels wide (close ... far away) and placed
in a 1280x720 frame. Then the CCTV face detector runs on the whole frame and the face is recognized
against all OTHER enrolled photos (leave-one-out: the photo is never matched against itself).

Results go to the screen and to data/cctv_pose_results.csv.
Usage: python evaluate_cctv.py
"""
import csv
from collections import defaultdict

import cv2
import numpy as np

import config
from enroll import PHOTO_EXT
from faces import detect_faces, load_face_app

POSES = ["front", "left", "right", "up", "down"]
FACE_WIDTHS = [160, 80, 48, 32, 24]
FRAME_W, FRAME_H = 1280, 720


def pose_of(name):
    n = name.lower()
    return next((p for p in POSES if p in n), "other")


def main():
    app = load_face_app()
    size = (config.CCTV_FACE_DET_SIZE, config.CCTV_FACE_DET_SIZE)
    photos = []   # (student, pose, image, face box, embedding)
    for person in sorted(p for p in config.ENROLLMENT_DIR.iterdir() if p.is_dir()):
        for f in sorted(x for x in person.iterdir() if x.suffix.lower() in PHOTO_EXT):
            img = cv2.imread(str(f))
            faces = detect_faces(app, img) if img is not None else []
            if not faces:
                continue
            face = max(faces, key=lambda q: q.bbox[2] - q.bbox[0])
            photos.append((person.name.lower(), pose_of(f.name), img, face.bbox, face.normed_embedding))
    gallery = np.array([p[4] for p in photos])
    names = [p[0] for p in photos]

    rows = []
    rng = np.random.default_rng(0)
    for i, (student, pose, img, box, _) in enumerate(photos):
        x1, y1, x2, y2 = box
        fw = x2 - x1
        for width in FACE_WIDTHS:
            s = width / fw
            small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            sh, sw = small.shape[:2]
            if sw > FRAME_W or sh > FRAME_H:   # crop around the face so it fits the frame
                cx, cy = int((x1 + x2) / 2 * s), int((y1 + y2) / 2 * s)
                ax, ay = max(0, min(sw - FRAME_W, cx - FRAME_W // 2)), max(0, min(sh - FRAME_H, cy - FRAME_H // 2))
                small = small[ay:ay + min(sh, FRAME_H), ax:ax + min(sw, FRAME_W)]
                sh, sw = small.shape[:2]
            frame = np.full((FRAME_H, FRAME_W, 3), 90, np.uint8)
            ox, oy = int(rng.integers(0, FRAME_W - sw + 1)), int(rng.integers(0, FRAME_H - sh + 1))
            frame[oy:oy + sh, ox:ox + sw] = small
            bboxes, kpss = app.det_model.detect(frame, input_size=size, max_num=0)
            outcome, score = "no face found", 0.0
            if len(bboxes):
                k = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]))
                from insightface.app.common import Face
                face = Face(bbox=bboxes[k][:4], kps=kpss[k], det_score=bboxes[k][4])
                app.models["recognition"].get(frame, face)
                sims = gallery @ face.normed_embedding
                sims[i] = -1.0                         # leave-one-out: never match the photo itself
                best = int(np.argmax(sims))
                score = float(sims[best])
                if score < config.MATCH_THRESHOLD:
                    outcome = "unknown"
                else:
                    outcome = "correct" if names[best] == student else "wrong person"
            rows.append({"student": student, "pose": pose, "face_px": width, "outcome": outcome, "score": round(score, 3)})

    config.DATA_DIR.mkdir(exist_ok=True)
    out = config.DATA_DIR / "cctv_pose_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(photos)} photos of {len(set(names))} students; cells = correct / unknown / wrong / no face (out of n)")
    print(f"{'pose':>6} | " + " | ".join(f"{w:>3} px face".rjust(17) for w in FACE_WIDTHS))
    for pose in POSES + ["other"]:
        cells = []
        for w in FACE_WIDTHS:
            r = [x for x in rows if x["pose"] == pose and x["face_px"] == w]
            if not r:
                continue
            c = defaultdict(int)
            for x in r:
                c[x["outcome"]] += 1
            cells.append(f"{c['correct']}/{c['unknown']}/{c['wrong person']}/{c['no face found']} (n={len(r)})".rjust(17))
        if cells:
            print(f"{pose:>6} | " + " | ".join(cells))
    print("Saved", out)


if __name__ == "__main__":
    main()
