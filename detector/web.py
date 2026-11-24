"""Web app for the PC usage monitor: one camera per PC.

Each PC's camera page ("Station") captures its own camera in the browser, shows it live,
and sends frames here for detection; the results come back as boxes/labels drawn on top
of the live video. The Security console shows every PC and the incident queue.

Usage:
  python web.py                 # then open http://localhost:8765
"""
import argparse
import base64
import csv
import json
import os
import re
import shutil
import threading
import time

from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

import config
from detection import CCTV_PROFILE, UNIDENTIFIED, analyze, new_camera
from reid import Gallery
from drawing import blur_faces, describe, draw, draw_boxes
from enroll import PHOTO_EXT, enroll
import preprocess as pp
from faces import Recognizer, detect_faces, load_face_app
from incidents import Incidents
from room import ROOM_LOG, RoomTracker
from sightings import Sightings
from tracker import Tracker, load_assignments, save_assignments
from videotest import VideoTest
from photocheck import check_photo
from evaluate_accuracy import measure
from ultralytics import YOLO

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v", ".wmv"}

PC_NAME = re.compile(r"^[A-Za-z0-9_-]{1,20}$")
BUILD = str(int(time.time()))   # new on every start / rebuild: open pages then reload themselves


class NoRecognizer:
    """Used until someone is enrolled: every face is unknown."""
    names, embeddings = [], np.zeros((0, 512))

    def identify(self, embedding):
        return "unknown", 0.0


def load_recognizer():
    return Recognizer() if config.EMBEDDINGS_FILE.exists() else NoRecognizer()


def marker_time():
    try:
        return config.RESET_MARKER.stat().st_mtime
    except OSError:
        return None


def load_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


class Engine:
    """Holds the models, the rules tracker and the latest result from every PC camera."""

    def __init__(self):
        print("Loading models...")
        new_camera()                           # loads/downloads YOLO26 once at startup
        self.face_app = load_face_app()
        self.lock = threading.Lock()           # one frame at a time through the models
        self.t0 = time.monotonic()
        self.reset_mark = marker_time()        # reset.py touches this; then all memory below is cleared
        self.version = 0                       # +1 on every change to PCs, students, watchlist, incidents:
        self.load_data()                       #   open pages then reload the lists they show
        self.photo_yolo = None                 # YOLO26 for "Check a photo" (loaded on first use)
        self.video_face_app = None             # separate face models for video tests (loaded on first use)
        self.next_cleanup = 0.0
        self.cleanup()

    def load_data(self):
        """Everything that comes from the data folder or was learned while running (not the models)."""
        self.recognizer = load_recognizer()
        self.assignments = load_assignments()  # {"PC-01": ["juan", "maria"], ...}
        self.watchlist = load_json(config.WATCHLIST_FILE, [])
        self.tracker = Tracker(self.assignments, watchlist=self.watchlist)
        self.incidents = Incidents()
        self.stations = {}                     # pc -> latest result, thumbnail, timing
        self.room = RoomTracker(self.watchlist)  # CCTV cameras: who is in the room
        self.cctv = {}                         # CCTV camera name -> latest result, thumbnail, timing
        self.sightings = Sightings()           # shared by all cameras: who was last recognized where
        self.last_report = load_json(config.ENROLL_REPORT_FILE, None)   # photo quality + best photo per person
        self.video_tests = {}                  # id -> VideoTest (this session)
        self.stranger_cache = None

    def check_reset(self):
        """reset.py was run while the app is running: drop the old feed, timers and visits from memory
        (otherwise they stay on screen and get written back) and load the fresh data."""
        mark = marker_time()
        if mark == self.reset_mark:
            return
        with self.lock:
            if mark == self.reset_mark:
                return
            self.reset_mark = mark
            self.tracker.usage.open.clear()      # visits from before the reset are not logged
            self.load_data()
            self.version += 1
            print("Data was reset: cleared the feed, timers and visits in memory and reloaded the data folder.")

    def cleanup(self):
        """Data retention: delete incidents, snapshots and log rows older than RETENTION_DAYS (hourly)."""
        if not config.RETENTION_DAYS or time.monotonic() < self.next_cleanup:
            return
        self.next_cleanup = time.monotonic() + 3600
        removed = self.incidents.prune(config.RETENTION_DAYS)
        removed += prune_csv(config.SESSIONS_LOG, "start", config.RETENTION_DAYS)
        removed += prune_csv(config.ALERTS_LOG, "time", config.RETENTION_DAYS)
        removed += prune_csv(ROOM_LOG, "entered", config.RETENTION_DAYS)
        removed += prune_video_tests(config.VIDEO_TEST_KEEP, config.RETENTION_DAYS)
        if removed:
            print(f"Retention: removed {removed} item(s) older than {config.RETENTION_DAYS} days")

    def now(self):
        return time.monotonic() - self.t0

    def process(self, pc, frame):
        """Runs detection + rules for one frame from one PC's camera. Returns what to draw."""
        self.check_reset()
        with self.lock:
            if pc not in self.assignments:   # deleted (or never created): don't bring it back
                return None
            self.cleanup()
            t = self.now()
            started = time.monotonic()
            if pc not in self.stations:   # each PC camera gets its own YOLO26 tracker
                self.stations[pc] = {"thumb": None, "thumb_t": -1, "cam": new_camera(name=pc, sightings=self.sightings)}
            st = self.stations[pc]
            result = analyze(frame, st["cam"], self.face_app, self.recognizer)
            ms = round((time.monotonic() - started) * 1000)

            names = {p["name"] for p in result["people"] if p["name"] != UNIDENTIFIED
                     and (p["user"] or p["name"] in self.tracker.watchlist)}   # user + banned bystanders
            self.tracker.update(t, pc, names, result["persons"], result["covered"])
            self.tracker.check_liveness(t, pc, result["people"])
            self.tracker.log_people(t, pc, result["people"])
            people = describe(pc, result, self.tracker, t)

            new_alerts = self.tracker.take_new_alerts()
            # Privacy: everything an admin sees or that is stored gets the faces pixelated first.
            shown = blur_faces(frame.copy(), people) if config.PRIVACY_BLUR else frame
            for alert in new_alerts:   # evidence: fully labeled frame (blurred before it is ever saved)
                self.incidents.add(alert, shown, draw(shown.copy(), pc, people, result["covered"]), people)
            if t - st["thumb_t"] >= 1.0:   # camera wall: clean image, the page draws the labels on top
                clean = draw_boxes(shown.copy(), people)
                small = cv2.resize(clean, (480, round(clean.shape[0] * 480 / clean.shape[1])))
                st["thumb"] = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                st["thumb_t"] = t
            st.update({"t": t, "ms": ms, "covered": result["covered"], "persons": result["persons"],
                       "people": people, "size": [frame.shape[1], frame.shape[0]], "prep": result["prep"]})
            return {"pc": pc, "assigned": self.assignments.get(pc, []), "covered": result["covered"],
                    "covered_reason": result.get("covered_reason", ""), "prep": result["prep"],
                    "persons": result["persons"], "people": people, "ms": ms,
                    "covered_for": round(t - self.tracker.covered_since[pc]) if pc in self.tracker.covered_since else 0}

    def process_cctv(self, cam, frame):
        """Detection + room rules for one frame from a CCTV (room) camera: everyone in view is
        found and named, nobody is "the PC user"."""
        self.check_reset()
        with self.lock:
            self.cleanup()
            t = self.now()
            started = time.monotonic()
            if cam not in self.cctv:
                self.cctv[cam] = {"thumb": None, "thumb_t": -1,
                                  "cam": new_camera(CCTV_PROFILE, name=f"CCTV {cam}", sightings=self.sightings)}
            st = self.cctv[cam]
            result = analyze(frame, st["cam"], self.face_app, self.recognizer)
            ms = round((time.monotonic() - started) * 1000)
            self.room.update(t, cam, result["people"], result["covered"])
            people = self.room.describe(cam, result["people"], t)
            shown = blur_faces(frame.copy(), people) if config.PRIVACY_BLUR else frame
            for alert in self.room.take_new_alerts():
                self.incidents.add(alert, shown, draw(shown.copy(), f"CCTV {cam}", people, result["covered"]), people)
            if t - st["thumb_t"] >= 1.0:
                clean = draw_boxes(shown.copy(), people)
                small = cv2.resize(clean, (640, round(clean.shape[0] * 640 / clean.shape[1])))
                st["thumb"] = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                st["thumb_t"] = t
            st.update({"t": t, "ms": ms, "covered": result["covered"], "people": people})
            return {"camera": cam, "covered": result["covered"], "covered_reason": result.get("covered_reason", ""),
                    "persons": result["persons"], "people": people, "ms": ms, "prep": result["prep"],
                    "covered_for": round(t - self.room.covered_since[cam]) if cam in self.room.covered_since else 0}

    def cctv_overview(self):
        t = self.now()
        cams = []
        for cam, st in sorted(self.cctv.items()):
            online = self.room.online(cam, t)
            people = st.get("people", []) if online else []
            cams.append({"camera": cam, "online": online, "covered": cam in self.room.covered_since,
                         "ms": st.get("ms"), "has_thumb": bool(st.get("thumb")),
                         "people": [{k: p[k] for k in ("id", "display", "label", "kind", "how", "seconds", "name")}
                                    for p in people]})
        return {"cameras": cams, "in_room": self.room.in_room(t), "log": read_csv_tail(ROOM_LOG, 20),
                "unknown_sec": config.CCTV_UNKNOWN_SEC, "log_gap": config.CCTV_LOG_GAP}

    def pc_list(self):
        return sorted(set(self.assignments) | set(self.stations))

    def overview(self):
        t, tr = self.now(), self.tracker
        out = []
        for pc in self.pc_list():
            st = self.stations.get(pc, {})
            assigned = self.assignments.get(pc, [])
            user = next((p for p in st.get("people", []) if p["user"]), None) if tr.online(pc, t) else None
            crowd = tr.crowd_since.get(pc)
            seen = tr.last_assigned_seen.get(pc)
            # assigned student not at their PC but seen by another camera (CCTV or another PC) just now
            here = user["name"] if user else None
            elsewhere = []
            for name in assigned:
                if name == here:
                    continue
                w = self.sightings.where(name, t, config.ROOM_NOTE_SEC)
                if w and w["camera"] != pc:
                    elsewhere.append({"name": name, **w})
            out.append({
                "pc": pc, "assigned": assigned, "online": tr.online(pc, t),
                "covered": pc in tr.covered_since, "persons": st.get("persons", 0) if tr.online(pc, t) else 0,
                "ms": st.get("ms"), "user": user,
                "away": round(t - seen) if assigned and seen is not None else None,
                "absent_alert": ("ABSENT", pc, "-") in tr.fired,
                "crowd_seconds": round(t - crowd) if crowd is not None else 0,
                "has_thumb": bool(st.get("thumb")),
                "elsewhere": elsewhere,
            })
        return out


def read_csv_tail(path, n=15):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))[-n:][::-1]


def prune_csv(path, column, days):
    """Data retention for a log: drop rows whose timestamp column is older than `days`. Returns rows removed."""
    if not path.exists():
        return 0
    cutoff = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - days * 86400))
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    keep = [r for r in rows if not (r.get(column, "")[:4].isdigit() and r[column] < cutoff)]
    if len(keep) != len(rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(keep)
    return len(rows) - len(keep)


def video_test_dirs():
    root = config.VIDEO_TEST_DIR
    return sorted((d for d in root.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: d.name) \
        if root.exists() else []


def prune_video_tests(keep, days=0):
    """Keep only the newest `keep` video test results (and none older than `days`). Returns how many were removed."""
    dirs = video_test_dirs()
    cutoff_ms = (time.time() - days * 86400) * 1000 if days else 0
    old = dirs[:-keep] if keep and len(dirs) > keep else []
    old += [d for d in dirs if d not in old and int(d.name) < cutoff_ms]
    for d in old:
        test = engine.video_tests.get(d.name) if engine else None
        if test and test.state["status"] in ("queued", "running"):
            continue
        shutil.rmtree(d, ignore_errors=True)
        if engine:
            engine.video_tests.pop(d.name, None)
    return len(old)


def video_test_state(vid):
    """Live state of a running test, or the saved summary of a finished one."""
    test = engine.video_tests.get(vid)
    if test:
        return test.snapshot()
    f = config.VIDEO_TEST_DIR / vid / "summary.json"
    return json.loads(f.read_text()) if f.exists() else None


def enrolled_people():
    root = config.ENROLLMENT_DIR
    if not root.exists():
        return {}
    return {p.name: sorted(f.name for f in p.iterdir() if f.suffix.lower() in PHOTO_EXT)
            for p in sorted(root.iterdir()) if p.is_dir()}


def enrollment_changes(report):
    """Compares the enrollment/ folders with the last enrollment, so new students or added/removed
    photos are noticed automatically: [{"name", "status": new|changed|removed, "photos"}]."""
    on_disk = {name.lower(): {f.lower() for f in files} for name, files in enrolled_people().items()}
    done = {e["name"]: {p["file"].lower() for p in e["photos"]} for e in (report or [])}
    changes = []
    for name, files in on_disk.items():
        if not files:
            continue   # empty folder: nothing to enroll yet
        if name not in done:
            changes.append({"name": name, "status": "new", "photos": len(files)})
        elif files != done[name]:
            changes.append({"name": name, "status": "changed", "photos": len(files)})
    changes += [{"name": n, "status": "removed", "photos": None} for n in done if n not in on_disk]
    return changes


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.VIDEO_MAX_MB * 1024 * 1024
engine: Engine = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def state():
    engine.check_reset()
    t, tr = engine.now(), engine.tracker
    return jsonify({
        "clock": tr.stamp(t), "pcs": engine.overview(), "assignments": engine.assignments,
        "reset_id": str(engine.reset_mark or ""), "build": BUILD, "version": engine.version,   # changes when reset.py runs: the page then forgets its cameras
        "people": sorted(set(engine.recognizer.names)), "watchlist": engine.watchlist,
        "enroll_pending": enrollment_changes(engine.last_report),
        "incidents": engine.incidents.recent(), "open_incidents": engine.incidents.open_count(),
        "alerts_total": len(read_csv_tail(config.ALERTS_LOG, 10_000)),
        "config": {k: getattr(config, k) for k in [
            "MATCH_THRESHOLD", "MIN_FACE_PX", "PRIVACY_BLUR", "RETENTION_DAYS", "PRESENCE_GAP", "SESSION_GAP", "WATCHLIST_SEC", "TAMPER_SEC",
            "UNAUTHORIZED_SEC", "WRONG_SEAT_SEC", "ABSENT_SEC", "CROWD_COUNT", "CROWD_SEC"]},
    })


@app.route("/api/sessions")
def sessions():
    """The whole usage log (kept for RETENTION_DAYS) + the visits still in progress; the page filters it."""
    from usage import read_all
    t, tr = engine.now(), engine.tracker
    with engine.lock:
        active = tr.usage.active(t)
    return jsonify({"active": active, "rows": read_all(config.SESSIONS_LOG)[::-1],
                    "pcs": engine.pc_list(), "assignments": engine.assignments,
                    "session_gap": config.SESSION_GAP, "presence_gap": config.PRESENCE_GAP,
                    "retention_days": config.RETENTION_DAYS})


@app.route("/api/frame", methods=["POST"])
def frame_in():
    """One camera frame (JPEG) from a PC's station page -> detection results as JSON."""
    pc = request.args.get("pc", "")
    if not PC_NAME.match(pc):
        return jsonify({"error": "Choose which PC this camera belongs to."}), 400
    frame = cv2.imdecode(np.frombuffer(request.get_data(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "bad frame"}), 400
    out = engine.process(pc, frame)
    if out is None:
        return jsonify({"error": f"{pc} does not exist any more (it was deleted). Choose a PC for this camera.",
                        "missing_pc": pc}), 404
    return jsonify(out)


@app.route("/api/cctv/frame", methods=["POST"])
def cctv_frame_in():
    """One frame (JPEG) from a CCTV / room camera -> everyone in view, as JSON."""
    cam = request.args.get("cam", "")
    if not PC_NAME.match(cam):
        return jsonify({"error": "Give this CCTV camera a name (letters, numbers, - and _)."}), 400
    frame = cv2.imdecode(np.frombuffer(request.get_data(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "bad frame"}), 400
    return jsonify(engine.process_cctv(cam, frame))


@app.route("/api/cctv")
def cctv_state():
    return jsonify(engine.cctv_overview())


@app.route("/api/cctv/test")
def cctv_test():
    """Summary of evaluate_cctv.py (head pose x face size -> correct / unknown / wrong / no face)."""
    path = config.DATA_DIR / "cctv_pose_results.csv"
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    sizes = sorted({int(r["face_px"]) for r in rows}, reverse=True)
    key = {"correct": "correct", "unknown": "unknown", "wrong person": "wrong", "no face found": "noface"}
    table = []
    for pose in ["front", "left", "right", "up", "down", "other"]:
        cells = {}
        for w in sizes:
            sel = [r for r in rows if r["pose"] == pose and int(r["face_px"]) == w]
            if sel:
                c = {"correct": 0, "unknown": 0, "wrong": 0, "noface": 0, "n": len(sel)}
                for r in sel:
                    c[key[r["outcome"]]] += 1
                cells[w] = c
        if cells:
            table.append({"pose": pose, "n": max(c["n"] for c in cells.values()), "cells": cells})
    rate = {w: round(100 * sum(r["outcome"] == "correct" for r in rows if int(r["face_px"]) == w) /
                     max(1, sum(int(r["face_px"]) == w for r in rows))) for w in sizes}
    return jsonify({"sizes": sizes, "rows": table, "rate": rate, "total": len(rows),
                    "wrong_total": sum(r["outcome"] == "wrong person" for r in rows)})


@app.route("/api/cctv/<cam>/stopped", methods=["POST"])
def cctv_stopped(cam):
    """A CCTV camera was stopped, renamed or removed: drop it from the CCTV overview right away."""
    with engine.lock:
        engine.cctv.pop(cam, None)
        engine.room.last_frame.pop(cam, None)
        engine.room.covered_since.pop(cam, None)
    return jsonify({"ok": True})


@app.route("/api/cctv/<cam>/thumb")
def cctv_thumb(cam):
    st = engine.cctv.get(cam)
    if not st or not st.get("thumb"):
        abort(404)
    return Response(st["thumb"], mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/api/video_tests", methods=["POST"])
def video_test_start():
    """Upload a video and start testing it: form fields video, profile (cctv|pc), expect (names), fps."""
    file = request.files.get("video")
    ext = Path(file.filename or "").suffix.lower() if file else ""
    if not file or ext not in VIDEO_EXT:
        return jsonify({"error": "Choose a video file (" + ", ".join(sorted(VIDEO_EXT)) + ")."}), 400
    if any(t.state["status"] in ("queued", "running") for t in engine.video_tests.values()):
        return jsonify({"error": "A video is already being tested. Wait for it or stop it first."}), 409
    vid = str(int(time.time() * 1000))
    folder = config.VIDEO_TEST_DIR / vid
    folder.mkdir(parents=True)
    upload = folder / ("upload" + ext)
    file.save(upload)
    enrolled = set(map(str, engine.recognizer.names))
    expected = [n for n in request.form.get("expect", "").split(",") if n in enrolled]
    try:
        fps = max(0.0, min(30.0, float(request.form.get("fps", config.VIDEO_TEST_FPS))))
    except ValueError:
        fps = config.VIDEO_TEST_FPS
    if engine.video_face_app is None:
        engine.video_face_app = load_face_app()   # its own copy, so live cameras and the test never share a model
    test = VideoTest(upload, folder, engine.video_face_app, engine.recognizer,
                     profile="pc" if request.form.get("profile") == "pc" else "cctv",
                     expected=expected, fps_limit=fps, blur=config.PRIVACY_BLUR,
                     name=re.sub(r"[^\w .()-]", "_", Path(file.filename).name)[:80])
    engine.video_tests[vid] = test

    def work():
        test.run()
        upload.unlink(missing_ok=True)            # privacy: only the face-blurred result is kept
        prune_video_tests(config.VIDEO_TEST_KEEP)
        engine.version += 1                       # open pages update their list of tests
    threading.Thread(target=work, daemon=True).start()
    engine.version += 1
    return jsonify({"id": vid})


@app.route("/api/video_tests")
def video_test_list():
    out = []
    for d in reversed(video_test_dirs()):
        st = video_test_state(d.name)
        if st:
            out.append({"id": d.name, "name": st.get("name"), "status": st.get("status"), "created": st.get("created"),
                        "seconds": st.get("video", {}).get("seconds"), "profile": st.get("profile")})
    return jsonify(out)


@app.route("/api/video_tests/<vid>")
def video_test_get(vid):
    st = video_test_state(vid) if vid.isdigit() else None
    return jsonify(st) if st else (jsonify({"error": "not found"}), 404)


@app.route("/api/video_tests/<vid>/frame.jpg")
def video_test_frame(vid):
    test = engine.video_tests.get(vid)
    if not test or not test.frame_jpg:
        abort(404)
    return Response(test.frame_jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/api/video_tests/<vid>/result.webm")
def video_test_video(vid):
    if not vid.isdigit():
        abort(404)
    return send_from_directory(config.VIDEO_TEST_DIR / vid, "result.webm", mimetype="video/webm")


@app.route("/api/video_tests/<vid>/stop", methods=["POST"])
def video_test_stop(vid):
    test = engine.video_tests.get(vid)
    if test:
        test.cancel()
    return jsonify({"ok": bool(test)})


@app.route("/api/video_tests/<vid>", methods=["DELETE"])
def video_test_delete(vid):
    test = engine.video_tests.get(vid)
    if test and test.state["status"] in ("queued", "running"):
        return jsonify({"error": "Stop the test first."}), 409
    if vid.isdigit():
        shutil.rmtree(config.VIDEO_TEST_DIR / vid, ignore_errors=True)
        engine.video_tests.pop(vid, None)
        engine.version += 1
    return jsonify({"ok": True})


@app.route("/api/pc/<pc>/thumb")
def pc_thumb(pc):
    st = engine.stations.get(pc)
    if not st or not st.get("thumb"):
        abort(404)
    return Response(st["thumb"], mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/api/process_photo", methods=["POST"])
def process_photo():
    """Image processing lab: every course operation applied to one photo, with stats and histograms."""
    file = request.files.get("photo")
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR) if file else None
    if img is None:
        return jsonify({"error": "Could not read that image."}), 400
    scale = 480 / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    try:
        dark = float(request.form.get("darken", 1.0))
    except ValueError:
        dark = 1.0
    if dark < 1:   # optional: simulate a dim room to see what the corrections do
        img = pp.brightness_contrast(img, max(0.02, dark), 0)
    gray = pp.to_gray(img)
    steps = [
        ("Original", "Lesson: Image Data Formats", f"{img.shape[1]}x{img.shape[0]}, {img.dtype}, 3 channels (BGR)", img),
        ("Grayscale", "Y = 0.299R + 0.587G + 0.114B", "1 channel, L = 256 levels", gray),
        ("Brightness / contrast", "Lesson 1: g = a*f + b", "a chosen so the mean becomes 110",
         pp.METHODS["brightness"][1](img)),
        ("Digital negative", "Lesson 2: s = (L-1) - r", "used to detect a blinded camera", pp.negative(img)),
        ("Min-max normalization", "Lesson 2: s = (r - rmin)/(rmax - rmin) * (L-1)", "on the luma channel",
         pp.on_luma(img, pp.normalize)),
        ("Histogram equalization", "Lesson 3: s_k = (L-1) * CDF(r_k)", "on the luma channel",
         pp.on_luma(img, pp.equalize_gray)),
        ("CLAHE", "Beyond the course: adaptive, tile-based equalization", "clip limit 2.0, 8x8 tiles",
         pp.on_luma(img, pp.clahe)),
    ]
    out = []
    for title, formula, note, im in steps:
        data = base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
        out.append({"title": title, "formula": formula, "note": note, "image": f"data:image/jpeg;base64,{data}",
                    "stats": pp.stats(im), "hist": pp.histogram(im, 64)})
    _, report = pp.enhance(img)
    return jsonify({"steps": out, "auto": report, "method": config.ENHANCE_METHOD})


@app.route("/api/test_photo", methods=["POST"])
def test_photo():
    """Check a photo: the full pipeline on one photo, with the time per step and every measurement."""
    file = request.files.get("photo")
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR) if file else None
    if img is None:
        return jsonify({"error": "Could not read that image."}), 400
    with engine.lock:
        if engine.photo_yolo is None:
            engine.photo_yolo = YOLO(config.YOLO_MODEL)   # plain detection (no tracking) for single photos
            engine.photo_yolo.predict(np.zeros((64, 64, 3), np.uint8), verbose=False)   # warm-up: honest timings
        result = check_photo(img, engine.face_app, engine.photo_yolo, engine.recognizer)
    return jsonify(result)


@app.route("/api/accuracy")
def accuracy():
    """Recognition accuracy on the enrolled photos (+ any stranger photos in data/strangers/), see evaluate_accuracy.py."""
    rec = engine.recognizer
    if len(set(map(str, rec.names))) < 1 or len(rec.names) < 2:
        return jsonify({"error": "Enroll at least one student with two or more photos first."})
    folder = config.DATA_DIR / "strangers"
    strangers = None
    if folder.exists():
        files = sorted(str(p) for p in folder.rglob("*") if p.suffix.lower() in PHOTO_EXT)
        cache = getattr(engine, "stranger_cache", None)
        if cache is None or cache[0] != files:      # embeddings are computed once per set of photos
            embs = []
            with engine.lock:
                for f in files:
                    img = cv2.imread(f)
                    faces = detect_faces(engine.face_app, img) if img is not None else []
                    if faces:
                        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                        embs.append(face.normed_embedding)
            engine.stranger_cache = cache = (files, np.array(embs))
        strangers = cache[1] if len(cache[1]) else None
    return jsonify(measure(rec.embeddings, rec.names, strangers))


@app.route("/api/enrollment")
def enrollment():
    return jsonify({"people": enrolled_people(), "report": engine.last_report,
                    "embeddings": len(engine.recognizer.names)})


@app.route("/api/enroll", methods=["POST"])
def run_enroll():
    with engine.lock:
        report = enroll(engine.face_app)
        engine.recognizer = load_recognizer()
        for st in [*engine.stations.values(), *engine.cctv.values()]:
            st["cam"]["gallery"] = Gallery()   # recognize everyone again with the new photos
    engine.last_report = report
    engine.version += 1
    return jsonify({"report": report, "embeddings": len(engine.recognizer.names)})


@app.route("/thumb/<person>/<photo>")
def thumb(person, photo):
    if photo not in enrolled_people().get(person, []):
        abort(404)
    img = cv2.imread(str(config.ENROLLMENT_DIR / person / photo))
    scale = 220 / max(img.shape[:2])
    img = cv2.resize(img, None, fx=scale, fy=scale)
    return Response(cv2.imencode(".jpg", img)[1].tobytes(), mimetype="image/jpeg")


@app.route("/snapshots/<name>")
def snapshot(name):
    if not re.fullmatch(r"\d+_(closeup|scene)\.jpg", name):
        abort(404)
    return send_from_directory(config.SNAPSHOT_DIR, name)


@app.route("/api/incidents/<iid>", methods=["POST"])
def incident_status(iid):
    item = engine.incidents.set_status(iid, (request.get_json() or {}).get("status"))
    engine.version += 1
    return jsonify(item) if item else (jsonify({"error": "not found"}), 404)


@app.route("/api/watchlist", methods=["POST"])
def save_watchlist():
    names = sorted({n for n in (request.get_json() or {}).get("names", []) if isinstance(n, str) and n})
    config.WATCHLIST_FILE.write_text(json.dumps(names, indent=2))
    with engine.lock:
        engine.watchlist = names
        engine.tracker.watchlist = set(names)
        engine.room.watchlist = set(names)
        engine.version += 1
    return jsonify({"ok": True, "watchlist": names})


def apply_assignments(data):
    save_assignments(data)
    with engine.lock:
        engine.assignments = data
        engine.tracker.set_assignments(data, engine.now())
        engine.version += 1


@app.route("/api/pcs/<pc>", methods=["POST"])
def set_pc(pc):
    """Create a PC or set who is assigned to it: {"names": ["juan", "maria"]} (any enrolled people)."""
    if not PC_NAME.match(pc):
        return jsonify({"error": "PC names use letters, numbers, - and _ only."}), 400
    enrolled = set(map(str, engine.recognizer.names))
    names = sorted({n for n in (request.get_json() or {}).get("names", []) if n in enrolled})
    data = {**engine.assignments, pc: names}
    apply_assignments(data)
    return jsonify({"ok": True, "pc": pc, "assigned": names})


@app.route("/api/pcs/<pc>/stopped", methods=["POST"])
def pc_camera_stopped(pc):
    """The camera panel of this PC was stopped or removed: show it offline right away (not 10 s later)."""
    with engine.lock:
        engine.stations.pop(pc, None)
        engine.tracker.last_frame.pop(pc, None)
        engine.tracker.covered_since.pop(pc, None)
        engine.version += 1
    return jsonify({"ok": True})


@app.route("/api/pcs/<pc>", methods=["DELETE"])
def delete_pc(pc):
    data = {k: v for k, v in engine.assignments.items() if k != pc}
    apply_assignments(data)
    with engine.lock:
        engine.stations.pop(pc, None)          # its last picture and tracker: gone from the dashboard
        tr = engine.tracker
        for d in (tr.covered_since, tr.last_frame, tr.crowd_since):
            d.pop(pc, None)
        tr.presence = {k: v for k, v in tr.presence.items() if k[0] != pc}
    return jsonify({"ok": True})


def main():
    global engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PCMON_PORT", 8765)))
    # 127.0.0.1 = only this computer. Docker sets PCMON_HOST=0.0.0.0 inside the container; the
    # compose file still publishes the port on 127.0.0.1 only.
    ap.add_argument("--host", default=os.environ.get("PCMON_HOST", "127.0.0.1"))
    args = ap.parse_args()
    engine = Engine()
    print(f"Open http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
