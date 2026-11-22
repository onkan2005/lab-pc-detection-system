"""Enrollment: turn enrollment/<name>/*.jpg into face embeddings, and report bad photos.

Usage:  python enroll.py              (uses ../enrollment)
        python enroll.py path/to/dir
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import config
from faces import detect_faces

PHOTO_EXT = {".jpg", ".jpeg", ".png"}


def enroll(app, root=config.ENROLLMENT_DIR):
    """Builds embeddings for every person folder. Returns a report: list of per-person dicts."""
    root = Path(root)
    people = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    embeddings, names, report = [], [], []

    for person in people:
        name = person.name.lower()
        entry = {"name": name, "photos": []}
        for photo in sorted(f for f in person.iterdir() if f.suffix.lower() in PHOTO_EXT):
            img = cv2.imread(str(photo))
            faces = detect_faces(app, img) if img is not None else []
            if img is None:
                entry["photos"].append({"file": photo.name, "status": "skip", "note": "cannot read file"})
                continue
            if not faces:
                entry["photos"].append({"file": photo.name, "status": "skip",
                                        "note": "no face found (side view, looking too far down, or blocked)"})
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            width = int(face.bbox[2] - face.bbox[0])
            notes = []
            if len(faces) > 1:
                notes.append(f"{len(faces)} faces, used the largest")
            if face.det_score < 0.6:
                notes.append("low confidence, maybe blurry or angled")
            entry["photos"].append({"file": photo.name, "status": "warn" if notes else "ok",
                                    "note": "; ".join(notes), "width": width,
                                    "score": round(float(face.det_score), 2)})
            embeddings.append(face.normed_embedding)
            names.append(name)
        entry["used"] = sum(p["status"] != "skip" for p in entry["photos"])
        entry["best"] = best_photo(entry["photos"])
        report.append(entry)

    config.DATA_DIR.mkdir(exist_ok=True)
    if embeddings:
        np.savez(config.EMBEDDINGS_FILE, embeddings=np.array(embeddings), names=np.array(names))
    config.ENROLL_REPORT_FILE.write_text(json.dumps(report, indent=2))
    return report


def best_photo(photos):
    """The photo to show for a person: a front shot with the most confident face detection
    (front-named photos win, then detection confidence, then a bigger face)."""
    usable = [p for p in photos if p["status"] != "skip"]
    if not usable:
        return photos[0]["file"] if photos else None
    return max(usable, key=lambda p: ("front" in p["file"].lower(), p["status"] == "ok",
                                      p["score"], p["width"]))["file"]


def main():
    from faces import load_face_app
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else config.ENROLLMENT_DIR
    report = enroll(load_face_app(), root)
    if not report:
        sys.exit(f"No person folders found in {root}")
    for entry in report:
        print(f"\n== {entry['name']} ({len(entry['photos'])} photos)")
        for p in entry["photos"]:
            detail = f"face {p['width']}px wide, detection {p['score']:.2f}" if "width" in p else ""
            note = f"  ({p['note']})" if p["note"] and detail else p["note"]
            print(f"  [{p['status'].upper():4}] {p['file']}: {detail}{note}")
        print(f"  -> {entry['used']}/{len(entry['photos'])} photos usable")
        if entry["used"] < 5:
            print("  !! Fewer than 5 usable photos, add more front/slight-angle shots")
    total = sum(e["used"] for e in report)
    print(f"\nSaved {total} embeddings -> {config.EMBEDDINGS_FILE}" if total else "\nNo usable faces; nothing saved.")


if __name__ == "__main__":
    main()
