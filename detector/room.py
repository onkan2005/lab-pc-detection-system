"""Room rules for CCTV cameras (one camera watching the whole lab, many people, far away).

A PC camera answers "who is using THIS PC"; a CCTV camera answers "who is in the ROOM":
  - every recognized student gets a room visit: entered at, last seen, minutes in the room
    (written to data/room_log.csv when they have not been seen for CCTV_LOG_GAP seconds)
  - a person who turns away keeps their name through appearance re-ID (reid.py), and is counted
    as "kept by clothes" so the technician can see the name is not from a face right now
  - incidents: a watchlisted student in the room (WATCHLIST_SEC), an unknown person in the room
    for CCTV_UNKNOWN_SEC (UNKNOWN_IN_ROOM), and the camera blocked (TAMPER_SEC)
"""
import csv
from datetime import datetime, timedelta

import config
from detection import UNIDENTIFIED

ROOM_LOG = config.DATA_DIR / "room_log.csv"
UNNAMED = (UNIDENTIFIED, "unknown")
COLORS = {"ok": "#1f9d55", "checking": "#f08c00", "alert": "#e03131", "bystander": "#868e96"}


class RoomTracker:
    def __init__(self, watchlist=()):
        self.watchlist = set(watchlist)
        self.start_wall = datetime.now()
        self.visits = {}          # (cam, name) -> {"first": t, "last": t}   enrolled people in the room
        self.unknown = {}         # (cam, pid) -> {"first": t, "last": t}    unknown people, by re-ID person id
        self.covered_since = {}   # cam -> t
        self.last_frame = {}      # cam -> t
        self.fired = set()
        self.new_alerts = []
        if not ROOM_LOG.exists():
            with open(ROOM_LOG, "w", newline="") as f:
                csv.writer(f).writerow(["name", "camera", "entered", "left", "minutes"])

    def stamp(self, t):
        return (self.start_wall + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S")

    def online(self, cam, t):
        return t - self.last_frame.get(cam, -1e9) <= config.CAMERA_OFFLINE_SEC

    def _alert(self, t, kind, cam, name, detail, pid=None):
        key = (kind, cam, name)
        if key in self.fired:
            return
        self.fired.add(key)
        with open(config.ALERTS_LOG, "a", newline="") as f:
            csv.writer(f).writerow([self.stamp(t), kind, cam, name, detail])
        print("ALERT", f"[{self.stamp(t)}] {kind}: {detail}")
        self.new_alerts.append({"time": self.stamp(t), "type": kind, "pc": cam, "name": name,
                                "detail": detail, "pid": pid})

    def take_new_alerts(self):
        alerts, self.new_alerts = self.new_alerts, []
        return alerts

    def _log_visit(self, cam, name, v):
        with open(ROOM_LOG, "a", newline="") as f:
            csv.writer(f).writerow([name, cam, self.stamp(v["first"]), self.stamp(v["last"]),
                                    f"{(v['last'] - v['first']) / 60:.2f}"])

    def update(self, t, cam, people, covered):
        """people: from detection.analyze() for this CCTV camera."""
        self.last_frame[cam] = t
        if covered:
            since = self.covered_since.setdefault(cam, t)
            if t - since >= config.TAMPER_SEC:
                self._alert(t, "CAMERA_BLOCKED", cam, "-", f"CCTV {cam} blocked for {t - since:.0f}s (possible tampering)")
        else:
            self.covered_since.pop(cam, None)
            self.fired.discard(("CAMERA_BLOCKED", cam, "-"))

        for p in people:
            if p["name"] in UNNAMED:
                key, store = (cam, p["id"]), self.unknown
            else:
                key, store = (cam, p["name"]), self.visits
            v = store.get(key)
            if v is None and store is self.unknown:
                v = self._continue_unknown(cam, key, t, {q["id"] for q in people})
            if v is None or t - v["last"] > config.CCTV_LOG_GAP:
                if v is not None and store is self.visits:
                    self._log_visit(cam, p["name"], v)
                store[key] = v = {"first": t, "last": t}
                name = p["name"] if store is self.visits else f"unknown-{p['id']}"
                self.fired = {f for f in self.fired if f[1:] != (cam, name)}   # new visit: may alert again
            v["last"] = t
            stayed = t - v["first"]
            if p["name"] in self.watchlist and stayed >= config.WATCHLIST_SEC:
                self._alert(t, "WATCHLIST", cam, p["name"], f"{p['name']} (on watchlist) is in the room ({cam})", p["id"])
            elif p["name"] in UNNAMED and stayed >= config.CCTV_UNKNOWN_SEC:
                self._alert(t, "UNKNOWN_IN_ROOM", cam, f"unknown-{p['id']}",
                            f"an unknown person has been in the room ({cam}) for {stayed:.0f}s", p["id"])

        # visits that ended: write them to the room log
        for key, v in list(self.visits.items()):
            if t - v["last"] > config.CCTV_LOG_GAP:
                self._log_visit(key[0], key[1], v)
                del self.visits[key]
        for key, v in list(self.unknown.items()):
            if t - v["last"] > config.CCTV_LOG_GAP:
                del self.unknown[key]

    def _continue_unknown(self, cam, key, t, present):
        """In a crowded room YOLO loses people behind others, and an unknown person can come back as a
        "new" unknown person. Then the timer would restart and the UNKNOWN_IN_ROOM alert might never
        fire. So a new unknown person continues the timer of an unknown person who vanished from this
        camera in the last UNKNOWN_CONTINUE_SEC seconds (and is not in view right now)."""
        gone = [(k, v) for k, v in self.unknown.items()
                if k[0] == cam and k[1] not in present and 0 < t - v["last"] <= config.UNKNOWN_CONTINUE_SEC]
        if not gone:
            return None
        k, v = max(gone, key=lambda kv: kv[1]["last"])   # the one who vanished most recently
        del self.unknown[k]
        self.unknown[key] = v
        old, new = f"unknown-{k[1]}", f"unknown-{key[1]}"
        self.fired = {(f[0], f[1], new) if f[1:] == (cam, old) else f for f in self.fired}
        return v

    def describe(self, cam, people, t):
        """Labels for the CCTV page: everyone is shown (no PC user / bystander split)."""
        unnamed = sorted((p for p in people if p["name"] in UNNAMED), key=lambda p: p["box"][0])
        slot = {id(p): i + 1 for i, p in enumerate(unnamed)} if len(unnamed) > 1 else {}
        out = []
        for p in people:
            name = p["name"]
            n = f" {slot[id(p)]}" if id(p) in slot else ""
            if name in UNNAMED:
                v = self.unknown.get((cam, p["id"]))
                seconds = round(t - v["first"]) if v else 0
                alert = ("UNKNOWN_IN_ROOM", cam, f"unknown-{p['id']}") in self.fired
                who = f"Unknown person{n}"
                how = "face not visible" if name == UNIDENTIFIED else "face not enrolled"
                if alert:
                    kind, label = "alert", f"ALERT: {who} in the room"
                else:
                    kind, label = "checking", f"{who} - {how} ({seconds}s/{config.CCTV_UNKNOWN_SEC}s)"
            else:
                v = self.visits.get((cam, name))
                seconds = round(t - v["first"]) if v else 0
                who = name
                how = (f"kept by {p.get('clothes', '')} clothes" if p.get("reid")
                       else "face" if p.get("face_box") else "kept by tracking (face not visible now)")
                if name in self.watchlist:
                    kind, label = "alert", f"WATCHLIST: {name}"
                else:
                    kind, label = "ok", name + (f" (by {p.get('clothes', '')} clothes)" if p.get("reid") else "")
            out.append({**p, "label": label, "color": COLORS[kind], "kind": kind, "display": who,
                        "how": how, "seconds": seconds})
        return out

    def in_room(self, t):
        """Enrolled people seen recently, for the CCTV page's room list."""
        rows = []
        for (cam, name), v in self.visits.items():
            rows.append({"name": name, "camera": cam, "entered": self.stamp(v["first"]),
                         "minutes": round((v["last"] - v["first"]) / 60, 1),
                         "last_seen": round(t - v["last"]), "watchlist": name in self.watchlist})
        return sorted(rows, key=lambda r: r["last_seen"])

    def close(self):
        for (cam, name), v in self.visits.items():
            self._log_visit(cam, name, v)
        self.visits.clear()
