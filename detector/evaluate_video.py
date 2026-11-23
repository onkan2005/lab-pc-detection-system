"""Test on a real video from the command line (the web app has the same test: Students -> Check a video).

Runs the video through the same pipeline as the live app (videotest.py) and prints what happened:
how many people, faces found, identities created, names given and how (face / tracking / clothes),
"unknown in room" alerts, time per step, and - when you say who is really in the video - accuracy
with and without a visible face. Saves an annotated, face-blurred result.webm + summary.json.

A person in a public sample video is not enrolled, so they can be enrolled TEMPORARILY from a few
frames (--enroll NAME --enroll-frames 0,10,20). Nothing is saved: the real enrollment is not touched.

Usage:
  python evaluate_video.py ../test_videos/my_cctv_clip.mp4 --expect juan,maria     # your own recording
  python evaluate_video.py ../test_videos/faceocc2.webm --enroll tester --fps 0      # public sample, every frame
Options: --profile cctv|pc (default cctv), --fps N (frames analyzed per second of video, 0 = all),
         --no-blur, --out DIR (default: <video>_result/ next to the video)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

import config
from faces import load_face_app
from videotest import STEP_LABEL, VideoTest


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


def enroll_from_video(path, frames, name, face_app, rec):
    cap, used = cv2.VideoCapture(str(path)), 0
    for i in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        faces = face_app.get(img) if ok else []
        if faces:
            rec.add(name, max(faces, key=lambda f: f.bbox[2] - f.bbox[0]).normed_embedding)
            used += 1
    return used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--enroll", help="temporarily enroll the person in the video under this name")
    ap.add_argument("--enroll-frames", default="0,10,20,30,40", help="frame numbers whose face is used to enroll")
    ap.add_argument("--expect", help="comma-separated names that really are in the video (default: the --enroll name)")
    ap.add_argument("--people", type=int, help="how many different people really appear (to judge the identity count)")
    ap.add_argument("--profile", choices=["cctv", "pc"], default="cctv")
    ap.add_argument("--fps", type=float, default=config.VIDEO_TEST_FPS, help="frames analyzed per second of video (0 = all)")
    ap.add_argument("--no-blur", action="store_true", help="do not pixelate faces in the result video")
    ap.add_argument("--out", help="output folder (default: <video>_result/ next to the video)")
    args = ap.parse_args()

    video = Path(args.video)
    face_app, rec = load_face_app(), TestRecognizer()
    if args.enroll:
        used = enroll_from_video(video, [int(x) for x in args.enroll_frames.split(",")], args.enroll, face_app, rec)
        print(f"Temporarily enrolled '{args.enroll}' from {used} frame(s) (not saved)")
    expected = (args.expect or args.enroll or "").split(",")
    out = Path(args.out) if args.out else video.with_name(video.stem + "_result")
    test = VideoTest(video, out, face_app, rec, profile=args.profile, expected=expected,
                     fps_limit=args.fps, blur=not args.no_blur)
    test.run()
    st = test.snapshot()
    if st["status"] == "error":
        raise SystemExit(st["error"])
    v, s, tm = st["video"], st["stats"], st["timings"]
    print(f"{video.name}: {v['width']}x{v['height']}, {v['seconds']} s, analyzed {s['frames']} frames "
          f"({v['processed_fps']}/s of video) in {st['progress']['elapsed']} s")
    print(f"People: most at once {s['max_people']}, avg {s['avg_people']} per frame; face found in {s['face_found_pct']}% of person boxes")
    print(f"Tracker: {s['tracks']} tracks, {s['rejoined']} re-joined by clothes/body; identities created {s['identities']}"
          + (f" (really {args.people} people)" if args.people else ""))
    print(f"'Unknown person N': highest N {s['highest_n']}, label changed while in view {s['label_changes']} times")
    print(f"Alerts: {s['alert_count']} " + ", ".join(f"{a['type']}@{a['time']}s" for a in s["alerts"]))
    for n in s["names"]:
        print(f"  {n['name']:<14} {n['seconds']:6.1f} s in view ({n['first']}..{n['last']} s), named by {n['by']}"
              + ("" if n["expected"] in (None, True) else "   <- WRONG NAME"))
    if "accuracy" in s:
        for part, title in (("face", "face visible"), ("no_face", "NO face visible"), ("all", "all")):
            a = s["accuracy"][part]
            print(f"  {title:<16} {a['n']:5d} boxes | correct {a['correct']:5d} ({a['correct_pct']}%) | "
                  f"unknown {a['unknown']:4d} | WRONG NAME {a['wrong']:3d}")
    print("Time per analyzed frame: " + ", ".join(f"{STEP_LABEL[k]} {ms} ms" for k, ms in tm["per_frame"].items())
          + f" = {tm['total_per_frame']} ms; ArcFace ran {tm['arcface_runs']}x ({tm['arcface_each']} ms each)")
    print(f"Saved {out / 'result.webm'} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
