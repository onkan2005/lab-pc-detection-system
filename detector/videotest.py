"""Video test: run a recorded video through the same pipeline as the live app and measure it.

Used by the web app ("Check a video" on the Students page) and by evaluate_video.py (command line).

Every processed frame goes through: preprocessing (course lessons) -> YOLO26 + tracker -> re-ID by
clothes/body -> SCRFD face detection -> ArcFace recognition -> room rules. While it runs, the live
state (current frame, the numbers behind each decision, time per step, running totals) can be read
from `VideoTest.state`; at the end it holds the summary and an annotated, face-blurred WebM video.

The video is processed on its own clock (frame number / fps), so timers and re-ID windows behave as
they would live, and by default at the live rate (VIDEO_TEST_FPS frames per second of video).
"""
import json
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

import config
from detection import CCTV_PROFILE, PC_PROFILE, UNIDENTIFIED, analyze, new_camera
from drawing import blur_faces
from room import RoomTracker

UNNAMED = (UNIDENTIFIED, "unknown")
STEPS = ("preprocess", "yolo", "reid", "face_detect", "liveness", "arcface")
STEP_LABEL = {"preprocess": "Preprocessing (lessons 1-3)", "yolo": "YOLO26 + tracker",
              "reid": "Re-ID (clothes/body)", "face_detect": "SCRFD face detection", "liveness": "Blink check (106 landmarks)",
              "arcface": "ArcFace recognition"}
BGR = {"ok": (85, 157, 31), "checking": (0, 140, 240), "alert": (49, 49, 224), "bystander": (150, 142, 134)}
OUT_WIDTH = 960      # result video / live frame width (smaller = smaller files)


class VideoTest:
    def __init__(self, src, out_dir, face_app, recognizer, profile="cctv", expected=(), fps_limit=None,
                 blur=True, name=None):
        self.src, self.out_dir = Path(src), Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.face_app, self.recognizer = face_app, recognizer
        self.profile = profile
        self.expected = {e for e in expected if e}
        self.fps_limit = config.VIDEO_TEST_FPS if fps_limit is None else fps_limit
        self.blur = blur
        self.cancelled = False
        self.frame_jpg = None
        self.lock = threading.Lock()
        self.state = {"status": "queued", "name": name or self.src.name, "profile": profile,
                      "expected": sorted(self.expected), "created": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---------- running ----------
    def start(self):
        threading.Thread(target=self.run, daemon=True).start()

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            self._run()
        except Exception as e:                      # shown on the page instead of failing silently
            self._set(status="error", error=f"{type(e).__name__}: {e}")
        finally:
            self._save_summary()

    def _set(self, **kw):
        with self.lock:
            self.state.update(kw)

    def _run(self):
        cap = cv2.VideoCapture(str(self.src))
        if not cap.isOpened():
            raise ValueError("could not open the video (unsupported format?)")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, round(fps / self.fps_limit)) if self.fps_limit else 1
        out_fps = fps / step
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # result video / live frame: at most OUT_WIDTH wide, and small CCTV videos (e.g. 360 px) are
        # enlarged to 720 px so the labels fit
        scale = min(OUT_WIDTH, max(w, 720)) / max(1, w)
        ow, oh = round(w * scale) // 2 * 2, round(h * scale) // 2 * 2
        writer = cv2.VideoWriter(str(self.out_dir / "result.webm"), cv2.VideoWriter_fourcc(*"VP80"), out_fps, (ow, oh))

        cam = new_camera(CCTV_PROFILE if self.profile == "cctv" else PC_PROFILE)
        gallery = cam["gallery"]
        created = []
        orig_new = gallery.new_person

        def counting_new(*a, **k):
            created.append(1)
            return orig_new(*a, **k)
        gallery.new_person = counting_new
        room = RoomTracker(room_log=self.out_dir / "room_log.csv", alerts_log=self.out_dir / "alerts.csv", quiet=True)

        self._set(status="running", video={"width": w, "height": h, "fps": round(fps, 2), "frames": total,
                                            "seconds": round(total / fps, 1), "every": step,
                                            "processed_fps": round(out_fps, 2)})
        timings = {}
        acc = {"frames": 0, "boxes": 0, "face_boxes": 0, "max_people": 0, "tracks": set(), "rejoined": 0,
               "covered_frames": 0, "alerts": [], "label_changes": 0, "highest_n": 0}
        names = defaultdict(lambda: {"frames": 0, "first": None, "last": None, "by": Counter()})
        verdict = {"face": Counter(), "no_face": Counter()}
        last_label = {}
        started = time.perf_counter()
        i = -1
        while not self.cancelled:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % step:
                continue
            t = i / fps
            result = analyze(frame, cam, self.face_app, self.recognizer, now=t, timings=timings)
            people = result["people"]
            room.update(t, "VIDEO", people, result["covered"])
            for a in room.take_new_alerts():
                acc["alerts"].append({"time": round(t, 1), "type": a["type"], "detail": a["detail"]})
            described = room.describe("VIDEO", people, t)

            # ---- running statistics ----
            acc["frames"] += 1
            acc["covered_frames"] += bool(result["covered"])
            acc["boxes"] += len(people)
            acc["max_people"] = max(acc["max_people"], len(people))
            for p in described:
                acc["face_boxes"] += bool(p["face_box"])
                if p["track"] is not None and p["track"] not in acc["tracks"]:
                    acc["tracks"].add(p["track"])
                    acc["rejoined"] += bool(p["reid"])
                how = "face" if p["face_box"] and not p["reid"] else "clothes" if p["reid"] else "tracking"
                if p["name"] not in UNNAMED:
                    n = names[p["name"]]
                    n["frames"] += 1
                    n["first"] = t if n["first"] is None else n["first"]
                    n["last"] = t
                    n["by"][how] += 1
                else:
                    last_n = p["display"].split()[-1]
                    acc["highest_n"] = max(acc["highest_n"], int(last_n) if last_n.isdigit() else 1)
                if self.expected:
                    v = "correct" if p["name"] in self.expected else "unknown" if p["name"] in UNNAMED else "wrong"
                    verdict["face" if p["face_box"] else "no_face"][v] += 1
                if p["id"] in last_label and last_label[p["id"]] != p["display"] and p["name"] in UNNAMED \
                        and last_label[p["id"]].startswith("Unknown"):
                    acc["label_changes"] += 1
            last_label = {p["id"]: p["display"] for p in described}

            # ---- annotated frame (faces pixelated first) ----
            shown = blur_faces(frame.copy(), described) if self.blur else frame.copy()
            if scale != 1:
                shown = cv2.resize(shown, (ow, oh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            self._annotate(shown, described, scale, t, result)
            writer.write(cv2.resize(shown, (ow, oh)) if shown.shape[1] != ow or shown.shape[0] != oh else shown)
            jpg = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

            elapsed = time.perf_counter() - started
            per_frame = elapsed / acc["frames"]
            left = max(0, (total - i - 1) // step) if total else 0
            with self.lock:
                self.frame_jpg = jpg
                self.state.update({
                    "progress": {"frame": i + 1, "frames": total, "time": round(t, 1),
                                 "pct": round(100 * (i + 1) / total, 1) if total else None,
                                 "elapsed": round(elapsed, 1), "eta": round(left * per_frame),
                                 "speed": round(acc["frames"] / elapsed, 2) if elapsed else None},
                    "current": self._current(described, result, t),
                    "timings": self._timings(timings, acc["frames"]),
                    "stats": self._stats(acc, created, names, verdict),
                })
        cap.release()
        writer.release()
        room.close()
        with self.lock:
            if not total and i >= 0:          # some formats (MPEG) do not say how many frames they have
                self.state["video"].update(frames=i + 1, seconds=round((i + 1) / fps, 1))
                if "progress" in self.state:
                    self.state["progress"]["frames"] = i + 1
            if not self.cancelled and "progress" in self.state:
                self.state["progress"].update(pct=100.0, eta=0)
        self._set(status="cancelled" if self.cancelled else "done", finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                  result_video="result.webm")

    # ---------- pieces of the live state ----------
    def _current(self, described, result, t):
        """The numbers behind every decision in the current frame (shown live on the page)."""
        prep = result["prep"]
        return {
            "time": round(t, 1), "covered": result["covered"], "covered_reason": result.get("covered_reason", ""),
            "prep": {"brightness": prep["before"]["brightness"], "contrast": prep["before"]["contrast"],
                     "after": [prep["after"]["brightness"], prep["after"]["contrast"]], "steps": prep["steps"],
                     "yolo_imgsz": prep.get("yolo_imgsz")},
            "people": [{"label": p["display"], "kind": p["kind"], "name": p["name"], "conf": p.get("conf"),
                        "face_score": p.get("face_score"), "checked": p.get("checked"),
                        "score": p["score"], "reid": p["reid"], "reid_sim": p.get("reid_sim"),
                        "how": p["how"], "seconds": p["seconds"], "clothes": p.get("clothes"),
                        "liveness": p.get("liveness"), "spoof": p.get("spoof"), "hint": p.get("hint"),
                        "face_px": (p["face_box"][2] - p["face_box"][0]) if p["face_box"] else None}
                       for p in described],
        }

    @staticmethod
    def _timings(timings, frames):
        n = max(1, frames)
        per = {k: round(timings.get(k, 0.0) / n, 1) for k in STEPS}
        return {"per_frame": per, "labels": STEP_LABEL, "total_per_frame": round(sum(per.values()), 1),
                "arcface_runs": timings.get("arcface_runs", 0),
                "arcface_each": round(timings.get("arcface", 0.0) / max(1, timings.get("arcface_runs", 0)), 1)}

    def _stats(self, acc, created, names, verdict):
        s = {"frames": acc["frames"], "boxes": acc["boxes"], "max_people": acc["max_people"],
             "avg_people": round(acc["boxes"] / max(1, acc["frames"]), 2),
             "face_found_pct": round(100 * acc["face_boxes"] / max(1, acc["boxes"]), 1),
             "tracks": len(acc["tracks"]), "rejoined": acc["rejoined"], "identities": len(created),
             "covered_frames": acc["covered_frames"], "alerts": acc["alerts"][-20:],
             "alert_count": len(acc["alerts"]), "label_changes": acc["label_changes"], "highest_n": acc["highest_n"],
             "names": sorted(({"name": k, "seconds": round(v["frames"] * self._frame_sec(), 1), "frames": v["frames"],
                               "first": round(v["first"], 1), "last": round(v["last"], 1), "by": dict(v["by"]),
                               "expected": (k in self.expected) if self.expected else None}
                              for k, v in names.items()), key=lambda r: -r["frames"])}
        if self.expected:
            def part(c):
                n = sum(c.values())
                return {"n": n, "correct": c["correct"], "unknown": c["unknown"], "wrong": c["wrong"],
                        "correct_pct": round(100 * c["correct"] / n, 1) if n else None}
            s["accuracy"] = {"face": part(verdict["face"]), "no_face": part(verdict["no_face"]),
                             "all": part(verdict["face"] + verdict["no_face"])}
        return s

    def _frame_sec(self):
        v = self.state.get("video", {})
        return 1 / v["processed_fps"] if v.get("processed_fps") else 0

    @staticmethod
    def _annotate(img, described, scale, t, result):
        for p in described:
            color = BGR.get(p["kind"], BGR["bystander"])
            x1, y1, x2, y2 = (round(v * scale) for v in p["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            if p["face_box"]:
                fx1, fy1, fx2, fy2 = (round(v * scale) for v in p["face_box"])
                cv2.rectangle(img, (fx1, fy1), (fx2, fy2), color, 1)
            how = "face" if p["face_box"] and not p["reid"] else "by clothes" if p["reid"] else "tracking"
            label = f"{p['display']} [{how}]"
            ty = y1 if y1 >= 18 else y1 + 18
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, ty - 18), (x1 + tw + 8, ty), color, -1)
            cv2.putText(img, label, (x1 + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        h = img.shape[0]
        cv2.rectangle(img, (0, h - 22), (img.shape[1], h), (0, 0, 0), -1)
        text = f"{int(t // 60):02d}:{t % 60:04.1f}  people in view: {len(described)}" + \
               ("  CAMERA BLOCKED" if result["covered"] else "")
        cv2.putText(img, text, (8, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # ---------- results ----------
    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))

    def _save_summary(self):
        with self.lock:
            (self.out_dir / "summary.json").write_text(json.dumps(self.state, indent=2))
