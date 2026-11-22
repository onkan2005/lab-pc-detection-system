"""Experiment: which preprocessing (course lessons) helps YOLO26 + face recognition in low light?

A webcam-style frame of each enrolled student is darkened to several light levels (with sensor
noise, like a dim room). Every preprocessing method is applied, then YOLO26 person detection,
face detection inside the YOLO box and ArcFace recognition are run. Results go to the screen
and to data/lighting_results.csv (use the table in the manuscript).

Usage: python evaluate_lighting.py
"""
import csv
import json

import cv2
import numpy as np
from ultralytics import YOLO

import config
import preprocess as pp
from detection import find_face
from enroll import PHOTO_EXT
from faces import Recognizer, detect_faces, load_face_app

LIGHT_LEVELS = [1.0, 0.45, 0.3, 0.2, 0.12, 0.08, 0.05, 0.035, 0.025]
METHODS = {"none": lambda img: img, **{k: fn for k, (_, fn) in pp.METHODS.items()}}
NOISE = 3.0   # sensor noise (std dev of intensity)


def webcam_frame(photo):
    """Head-and-shoulders photo placed in a 640x360 frame, like a PC webcam."""
    img = cv2.resize(cv2.imread(str(photo)), (340, 340))
    frame = np.full((360, 640, 3), 95, np.uint8)
    frame[10:350, 150:490] = img
    return frame


def main():
    yolo, app, rec = YOLO(config.YOLO_MODEL), load_face_app(), Recognizer()
    people = [d for d in sorted(config.ENROLLMENT_DIR.iterdir()) if d.is_dir()]
    rows = []
    report = json.loads(config.ENROLL_REPORT_FILE.read_text()) if config.ENROLL_REPORT_FILE.exists() else []
    best = {e["name"]: e.get("best") for e in report}
    for person in people:
        # the person's best photo from enrollment; otherwise any photo with a detectable face
        photos = sorted(p for p in person.iterdir() if p.suffix.lower() in PHOTO_EXT)
        photo = next((p for p in photos if p.name == best.get(person.name.lower())), None) or \
            next((p for p in photos if detect_faces(app, cv2.imread(str(p)))), None)
        if photo is None:
            print(f"skip {person.name}: no photo with a detectable face")
            continue
        frame = webcam_frame(photo)
        for k in LIGHT_LEVELS:
            rng = np.random.default_rng(0)
            dark = np.clip(frame.astype(np.float32) * k + rng.normal(0, NOISE, frame.shape), 0, 255).astype(np.uint8)
            s = pp.stats(dark)
            for method, fn in METHODS.items():
                img = fn(dark)
                boxes = yolo(img, conf=config.YOLO_CONF, imgsz=config.YOLO_IMGSZ, classes=[0], verbose=False)[0].boxes
                yolo_conf = float(boxes.conf.max()) if len(boxes) else 0.0
                name, score = "no face", 0.0
                if len(boxes):
                    face = find_face(img, boxes.xyxy[0].tolist(), app)
                    if face is not None:
                        app.models["recognition"].get(img, face)
                        name, score = rec.identify(face.normed_embedding)
                rows.append({"student": person.name, "light_%": round(k * 100, 1), "brightness": s["brightness"],
                             "contrast": s["contrast"], "method": method, "yolo_person_conf": round(yolo_conf, 2),
                             "recognized_as": str(name), "similarity": round(float(score), 2),
                             "correct": str(name) == person.name.lower()})

    config.DATA_DIR.mkdir(exist_ok=True)
    out = config.DATA_DIR / "lighting_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'light':>6} {'bright':>6} | " + " | ".join(f"{m:>12}" for m in METHODS))
    for k in LIGHT_LEVELS:
        cells = []
        for m in METHODS:
            r = [x for x in rows if x["light_%"] == round(k * 100, 1) and x["method"] == m]
            ok = sum(x["correct"] for x in r)
            sim = np.mean([x["similarity"] for x in r])
            cells.append(f"{ok}/{len(r)} {sim:5.2f}".rjust(12))
        b = np.mean([x["brightness"] for x in rows if x["light_%"] == round(k * 100, 1)])
        print(f"{k * 100:5.1f}% {b:6.1f} | " + " | ".join(cells))
    print("cells = correctly recognized / students, mean similarity")
    print(f"Saved {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
