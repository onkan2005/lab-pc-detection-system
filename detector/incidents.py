"""Incident queue for the lab technician.

Every alert becomes an incident with evidence snapshots. The technician moves it through:
  open -> dispatched (technician sent) -> resolved / false_alarm
"""
import json
import threading
import time

import cv2

import config

STATUSES = {"open", "dispatched", "resolved", "false_alarm"}


class Incidents:
    def __init__(self):
        self.lock = threading.Lock()
        config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        self.items = json.loads(config.INCIDENTS_FILE.read_text()) if config.INCIDENTS_FILE.exists() else []

    def _save(self):
        config.INCIDENTS_FILE.write_text(json.dumps(self.items, indent=2))

    def add(self, alert, raw_frame, annotated, people):
        """Store an alert with a close-up of the person (if any) and the full annotated camera view."""
        iid = f"{int(time.time() * 1000)}"
        files = {}
        pid = alert.get("pid")
        person = next((p for p in people if (pid is not None and p["id"] == pid) or p["name"] == alert["name"]), None)
        if person is not None:
            h, w = raw_frame.shape[:2]
            x1, y1, x2, y2 = person["face_box"] or person["box"]   # face if visible, else the YOLO person box
            pad = int((x2 - x1) * 0.5) if person["face_box"] else 0
            crop = raw_frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
            if crop.size:
                files["closeup"] = f"{iid}_closeup.jpg"
                cv2.imwrite(str(config.SNAPSHOT_DIR / files["closeup"]), crop)
        files["scene"] = f"{iid}_scene.jpg"
        cv2.imwrite(str(config.SNAPSHOT_DIR / files["scene"]), annotated)
        item = {"id": iid, **alert, "status": "open", "status_time": None, **files}
        with self.lock:
            self.items.append(item)
            self._save()
        return item

    def set_status(self, iid, status):
        if status not in STATUSES:
            return None
        with self.lock:
            for item in self.items:
                if item["id"] == iid:
                    item["status"] = status
                    item["status_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    self._save()
                    return item
        return None

    def recent(self, n=30):
        with self.lock:
            return list(reversed(self.items[-n:]))

    def open_count(self):
        with self.lock:
            return sum(i["status"] in ("open", "dispatched") for i in self.items)

    def prune(self, days):
        """Data retention: delete finished incidents (and their snapshots) older than `days`.
        Open / in-progress incidents are kept until the technician closes them."""
        cutoff_ms = (time.time() - days * 86400) * 1000
        with self.lock:
            old = [i for i in self.items if int(i["id"]) < cutoff_ms and i["status"] in ("resolved", "false_alarm")]
            for item in old:
                for key in ("closeup", "scene"):
                    if item.get(key):
                        (config.SNAPSHOT_DIR / item[key]).unlink(missing_ok=True)
            if old:
                self.items = [i for i in self.items if i not in old]
                self._save()
            return len(old)
