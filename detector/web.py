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
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

import config
from detection import UNIDENTIFIED, analyze, new_camera
from drawing import describe, draw, draw_boxes
from enroll import PHOTO_EXT, enroll
import preprocess as pp
from faces import Recognizer, detect_faces, load_face_app
from incidents import Incidents
from tracker import Tracker, load_assignments, save_assignments

PC_NAME = re.compile(r"^[A-Za-z0-9_-]{1,20}$")


class NoRecognizer:
    """Used until someone is enrolled: every face is unknown."""
    names, embeddings = [], np.zeros((0, 512))

    def identify(self, embedding):
        return "unknown", 0.0


def load_recognizer():
    return Recognizer() if config.EMBEDDINGS_FILE.exists() else NoRecognizer()


def load_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


class Engine:
    """Holds the models, the rules tracker and the latest result from every PC camera."""

    def __init__(self):
        print("Loading models...")
        new_camera()                           # loads/downloads YOLO26 once at startup
        self.face_app = load_face_app()
        self.recognizer = load_recognizer()
        self.lock = threading.Lock()           # one frame at a time through the models
        self.t0 = time.monotonic()
        self.assignments = load_assignments()  # {"PC-01": ["juan", "maria"], ...}
        self.watchlist = load_json(config.WATCHLIST_FILE, [])
        self.tracker = Tracker(self.assignments, watchlist=self.watchlist)
        self.incidents = Incidents()
        self.stations = {}                     # pc -> latest result, thumbnail, timing
        self.last_report = load_json(config.ENROLL_REPORT_FILE, None)   # photo quality + best photo per person

    def now(self):
        return time.monotonic() - self.t0

    def process(self, pc, frame):
        """Runs detection + rules for one frame from one PC's camera. Returns what to draw."""
        with self.lock:
            t = self.now()
            started = time.monotonic()
            if pc not in self.stations:   # each PC camera gets its own YOLO26 tracker
                self.stations[pc] = {"thumb": None, "thumb_t": -1, "cam": new_camera()}
            st = self.stations[pc]
            result = analyze(frame, st["cam"], self.face_app, self.recognizer)
            ms = round((time.monotonic() - started) * 1000)

            names = {p["name"] for p in result["people"] if p["name"] != UNIDENTIFIED
                     and (p["user"] or p["name"] in self.tracker.watchlist)}   # user + banned bystanders
            self.tracker.update(t, pc, names, result["persons"], result["covered"])
            people = describe(pc, result, self.tracker, t)

            new_alerts = self.tracker.take_new_alerts()
            for alert in new_alerts:   # evidence: fully labeled frame
                self.incidents.add(alert, frame, draw(frame.copy(), pc, people, result["covered"]), people)
            if t - st["thumb_t"] >= 1.0:   # camera wall: clean image, the page draws the labels on top
                clean = draw_boxes(frame.copy(), people)
                small = cv2.resize(clean, (480, round(clean.shape[0] * 480 / clean.shape[1])))
                st["thumb"] = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                st["thumb_t"] = t
            st.update({"t": t, "ms": ms, "covered": result["covered"], "persons": result["persons"],
                       "people": people, "size": [frame.shape[1], frame.shape[0]], "prep": result["prep"]})
            return {"pc": pc, "assigned": self.assignments.get(pc, []), "covered": result["covered"],
                    "covered_reason": result.get("covered_reason", ""), "prep": result["prep"],
                    "persons": result["persons"], "people": people, "ms": ms,
                    "covered_for": round(t - self.tracker.covered_since[pc]) if pc in self.tracker.covered_since else 0}

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
            out.append({
                "pc": pc, "assigned": assigned, "online": tr.online(pc, t),
                "covered": pc in tr.covered_since, "persons": st.get("persons", 0) if tr.online(pc, t) else 0,
                "ms": st.get("ms"), "user": user,
                "away": round(t - seen) if assigned and seen is not None else None,
                "absent_alert": ("ABSENT", pc, "-") in tr.fired,
                "crowd_seconds": round(t - crowd) if crowd is not None else 0,
                "has_thumb": bool(st.get("thumb")),
            })
        return out


def read_csv_tail(path, n=15):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))[-n:][::-1]


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
engine: Engine = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def state():
    t, tr = engine.now(), engine.tracker
    active = [{"name": n, "pc": pc, "minutes": round((s["last"] - s["start"]) / 60, 2),
               "assigned_here": tr.is_assigned(pc, n)} for (pc, n), s in tr.sessions.items()]
    return jsonify({
        "clock": tr.stamp(t), "pcs": engine.overview(), "assignments": engine.assignments,
        "people": sorted(set(engine.recognizer.names)), "watchlist": engine.watchlist,
        "enroll_pending": enrollment_changes(engine.last_report),
        "incidents": engine.incidents.recent(), "open_incidents": engine.incidents.open_count(),
        "active_sessions": active, "sessions": read_csv_tail(config.SESSIONS_LOG),
        "alerts_total": len(read_csv_tail(config.ALERTS_LOG, 10_000)),
        "config": {k: getattr(config, k) for k in [
            "MATCH_THRESHOLD", "MIN_FACE_PX", "PRESENCE_GAP", "SESSION_GAP", "WATCHLIST_SEC", "TAMPER_SEC",
            "UNAUTHORIZED_SEC", "WRONG_SEAT_SEC", "ABSENT_SEC", "CROWD_COUNT", "CROWD_SEC"]},
    })


@app.route("/api/frame", methods=["POST"])
def frame_in():
    """One camera frame (JPEG) from a PC's station page -> detection results as JSON."""
    pc = request.args.get("pc", "")
    if not PC_NAME.match(pc):
        return jsonify({"error": "Choose which PC this camera belongs to."}), 400
    frame = cv2.imdecode(np.frombuffer(request.get_data(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "bad frame"}), 400
    return jsonify(engine.process(pc, frame))


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
    file = request.files.get("photo")
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR) if file else None
    if img is None:
        return jsonify({"error": "Could not read that image."}), 400
    with engine.lock:
        faces = detect_faces(engine.face_app, img)
    rec = engine.recognizer
    out, results = img.copy(), []
    thick = max(2, img.shape[1] // 400)
    for i, face in enumerate(sorted(faces, key=lambda f: f.bbox[0]), 1):
        x1, y1, x2, y2 = map(int, face.bbox)
        name, score = rec.identify(face.normed_embedding)
        per_person = {}
        if len(rec.names):
            for n, s in zip(rec.names, rec.embeddings @ face.normed_embedding):
                per_person[str(n)] = round(max(per_person.get(str(n), -1.0), float(s)), 3)
        color = (0, 200, 0) if name != "unknown" else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        cv2.putText(out, f"#{i} {name}", (x1, max(y1 - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX, thick * 0.5, color, thick)
        results.append({"n": i, "name": str(name), "score": round(score, 3), "width": x2 - x1,
                        "det_score": round(float(face.det_score), 2),
                        "per_person": dict(sorted(per_person.items(), key=lambda kv: -kv[1]))})
    scale = 900 / max(out.shape[:2])
    if scale < 1:
        out = cv2.resize(out, None, fx=scale, fy=scale)
    data = base64.b64encode(cv2.imencode(".jpg", out)[1]).decode()
    return jsonify({"image": f"data:image/jpeg;base64,{data}", "faces": results,
                    "threshold": config.MATCH_THRESHOLD})


@app.route("/api/enrollment")
def enrollment():
    return jsonify({"people": enrolled_people(), "report": engine.last_report,
                    "embeddings": len(engine.recognizer.names)})


@app.route("/api/enroll", methods=["POST"])
def run_enroll():
    with engine.lock:
        report = enroll(engine.face_app)
        engine.recognizer = load_recognizer()
        for st in engine.stations.values():
            st["cam"]["ids"].clear()   # recognize everyone again with the new photos
    engine.last_report = report
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
    return jsonify(item) if item else (jsonify({"error": "not found"}), 404)


@app.route("/api/watchlist", methods=["POST"])
def save_watchlist():
    names = sorted({n for n in (request.get_json() or {}).get("names", []) if isinstance(n, str) and n})
    config.WATCHLIST_FILE.write_text(json.dumps(names, indent=2))
    with engine.lock:
        engine.watchlist = names
        engine.tracker.watchlist = set(names)
    return jsonify({"ok": True, "watchlist": names})


def apply_assignments(data):
    save_assignments(data)
    with engine.lock:
        engine.assignments = data
        engine.tracker.set_assignments(data, engine.now())


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


@app.route("/api/pcs/<pc>", methods=["DELETE"])
def delete_pc(pc):
    data = {k: v for k, v in engine.assignments.items() if k != pc}
    apply_assignments(data)
    with engine.lock:
        engine.stations.pop(pc, None)
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
