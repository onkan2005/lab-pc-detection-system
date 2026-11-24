"""Command-line version for one PC: opens that PC's camera in an OpenCV window.
(web.py is the main app; use this when running Python directly on a lab PC.)

Usage:
  python monitor.py --pc PC-01                  # camera 0 belongs to PC-01
  python monitor.py --pc PC-02 --source 1       # camera 1 belongs to PC-02
  python monitor.py --pc PC-01 --source test.mp4
Keys: q = quit
"""
import argparse
import json
import sys
import time

import cv2

import config
from detection import UNIDENTIFIED, analyze, new_camera
from drawing import describe, draw
from faces import Recognizer, load_face_app
from tracker import Tracker, load_assignments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc", required=True, help="which PC this camera belongs to, e.g. PC-01")
    ap.add_argument("--source", default="0", help="camera number or video file")
    args = ap.parse_args()

    assignments = load_assignments()
    watchlist = json.loads(config.WATCHLIST_FILE.read_text()) if config.WATCHLIST_FILE.exists() else []
    print("Loading models...")
    cam, face_app, recognizer = new_camera(), load_face_app(), Recognizer()

    live = args.source.isdigit()
    if live:   # DirectShow opens virtual cameras (Iriun, DroidCam) more reliably on Windows
        cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera/video: {args.source}")
    fps = 30 if live else (cap.get(cv2.CAP_PROP_FPS) or 30)

    tracker = Tracker(assignments, watchlist=watchlist, live=live)
    t0, frame_no = time.monotonic(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.monotonic() - t0 if live else frame_no / fps
        frame_no += 1
        result = analyze(frame, cam, face_app, recognizer)
        names = {p["name"] for p in result["people"] if p["name"] != UNIDENTIFIED
                 and (p["user"] or p["name"] in tracker.watchlist)}
        tracker.update(t, args.pc, names, result["persons"], result["covered"])
        tracker.check_liveness(t, args.pc, result["people"])
        tracker.log_people(t, args.pc, result["people"])
        tracker.take_new_alerts()   # printed and written to alerts.csv already
        cv2.imshow(f"{args.pc} camera", draw(frame, args.pc, describe(args.pc, result, tracker, t), result["covered"]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Logs: {config.SESSIONS_LOG} and {config.ALERTS_LOG}")


if __name__ == "__main__":
    main()
