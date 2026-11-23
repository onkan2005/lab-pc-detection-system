"""Test on a real video: does a person keep their name when their face is no longer visible?

Runs every frame of a video through the same pipeline as the live app (preprocessing -> YOLO26 +
ByteTrack -> re-ID by clothes/body -> face -> ArcFace) using the video's own clock, then reports, for
the frames where NO face could be seen, how often the person still had the right name.

The person in a public sample video is not enrolled, so they can be enrolled TEMPORARILY from a few
frames at the start (--enroll NAME --enroll-frames 0,10,20). Nothing is saved: the real enrollment,
embeddings and logs are not touched. Your enrolled students are still loaded, so the test also shows
whether a stranger is ever given one of your students' names.

Usage:
  python evaluate_video.py ../test_videos/faceocc2.webm --enroll tester --enroll-frames 0,10,20,30,40
  python evaluate_video.py ../test_videos/my_cctv_clip.mp4 --expect juan,maria     # your own recording
Options: --profile cctv|pc (default cctv), --no-blur, --out result.mp4
"""
import argparse
import csv
import types
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import config
import detection
from detection import CCTV_PROFILE, PC_PROFILE, UNIDENTIFIED, analyze, new_camera
from drawing import blur_faces
from faces import load_face_app

UNNAMED = (UNIDENTIFIED, "unknown")


class TestRecognizer:
    """Enrolled students (if any) + people enrolled temporarily from the video."""

    def __init__(self):
        if config.EMBEDDINGS_FILE.exists():
            data = np.load(config.EMBEDDINGS_FILE)
            self.embeddings, self.names = data["embeddings"], [str(n) for n in data["names"]]
        else:
            self.embeddings, self.names = np.zeros((0, 512), np.float32), []

    def add(self, name, embedding):
        self.embeddings = np.vstack([self.embeddings, embedding[None]])
        self.names.append(name)

    def identify(self, embedding):
        if not len(self.names):
            return "unknown", 0.0
        sims = self.embeddings @ embedding
        best = int(np.argmax(sims))
        return (self.names[best] if sims[best] >= config.MATCH_THRESHOLD else "unknown"), float(sims[best])


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames, fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--enroll", help="temporarily enroll the person in the video under this name")
    ap.add_argument("--enroll-frames", default="0,10,20,30,40", help="frame numbers whose face is used to enroll")
    ap.add_argument("--expect", help="comma-separated names that really are in the video (default: the --enroll name)")
    ap.add_argument("--profile", choices=["cctv", "pc"], default="cctv")
    ap.add_argument("--no-blur", action="store_true", help="do not pixelate faces in the result video")
    ap.add_argument("--out", help="annotated result video (default: <video>_result.mp4 next to the video)")
    args = ap.parse_args()

    video = Path(args.video)
    frames, fps = read_frames(video)
    if not frames:
        raise SystemExit(f"Could not read {video}")
    print(f"{video.name}: {len(frames)} frames, {fps:.0f} fps, {frames[0].shape[1]}x{frames[0].shape[0]}")

    face_app = load_face_app()
    rec = TestRecognizer()
    if args.enroll:
        used = 0
        for i in (int(x) for x in args.enroll_frames.split(",")):
            faces = face_app.get(frames[i])
            if faces:
                rec.add(args.enroll, max(faces, key=lambda f: f.bbox[2] - f.bbox[0]).normed_embedding)
                used += 1
        print(f"Temporarily enrolled '{args.enroll}' from {used} frame(s) (not saved)")
    expected = set((args.expect or args.enroll or "").split(",")) - {""}

    # Use the video's own clock, so "re-check every second" and re-ID time windows behave as they would live
    clock = types.SimpleNamespace(now=0.0)
    detection.time = types.SimpleNamespace(monotonic=lambda: clock.now)

    profile = CCTV_PROFILE if args.profile == "cctv" else PC_PROFILE
    cam = new_camera(profile)
    out_path = Path(args.out) if args.out else video.with_name(video.stem + "_result.mp4")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    rows, prev = [], None
    for i, frame in enumerate(frames):
        clock.now = i / fps
        result = analyze(frame, cam, face_app, rec)
        people = result["people"]
        main_p = max(people, key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1])) if people else None
        name = main_p["name"] if main_p else "-"
        face = bool(main_p and main_p["face_box"])
        how = "-" if not main_p else "face" if face and not main_p["reid"] else "clothes (re-ID)" if main_p["reid"] else "track"
        verdict = ("nobody" if not main_p else "correct" if name in expected
                   else "unknown" if name in UNNAMED else "WRONG NAME")
        rows.append({"frame": i, "time": round(i / fps, 2), "persons": len(people), "face_visible": face,
                     "name": name, "kept_by": how if not face else "face", "verdict": verdict})

        shown = blur_faces(frame.copy(), people) if not args.no_blur else frame.copy()
        for p in people:
            ok = p["name"] in expected
            color = (60, 180, 60) if ok else (60, 60, 220) if p["name"] not in UNNAMED else (150, 150, 150)
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(shown, (x1, y1), (x2, y2), color, 2)
            if p["face_box"]:
                fx1, fy1, fx2, fy2 = p["face_box"]
                cv2.rectangle(shown, (fx1, fy1), (fx2, fy2), color, 1)
            label = ("Unknown" if p["name"] in UNNAMED else p["name"]) + \
                    (" [face]" if p["face_box"] and not p["reid"] else " [no face: kept by " + ("clothes]" if p["reid"] else "track]"))
            ty = y1 if y1 >= 16 else y1 + 16       # box at the top edge: put the label inside it
            cv2.rectangle(shown, (x1, ty - 16), (x1 + 7 * len(label), ty), color, -1)
            cv2.putText(shown, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(shown, f"frame {i}  YOLO26 people: {len(people)}", (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 255), 1, cv2.LINE_AA)
        writer.write(shown)
        if name != prev and i:
            print(f"  frame {i:4d} ({i / fps:5.1f}s): {prev} -> {name}  (face visible: {face})")
        prev = name
    writer.release()

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    seen = [r for r in rows if r["verdict"] != "nobody"]
    with_face = [r for r in seen if r["face_visible"]]
    no_face = [r for r in seen if not r["face_visible"]]
    switches = sum(1 for a, b in zip(seen, seen[1:]) if a["name"] != b["name"])

    def line(title, rs):
        c = Counter(r["verdict"] for r in rs)
        n = max(1, len(rs))
        print(f"{title:<28} {len(rs):4d} frames | correct {c['correct']:4d} ({100 * c['correct'] / n:5.1f}%) | "
              f"unknown {c['unknown']:4d} | WRONG NAME {c['WRONG NAME']:3d}")

    print("\nResults (the largest person in each frame)")
    line("All frames with a person", seen)
    line("Face visible", with_face)
    line("NO face visible", no_face)
    kept = Counter(r["kept_by"] for r in no_face if r["verdict"] == "correct")
    print(f"Without a face the name was kept by: " + ", ".join(f"{k} {v}" for k, v in kept.items()))
    print(f"Name changes: {switches}")
    print(f"Saved {out_path} and {csv_path}")


if __name__ == "__main__":
    main()
