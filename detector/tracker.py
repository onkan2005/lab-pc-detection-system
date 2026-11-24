"""Usage logging and threshold-based alert rules, for many PCs (one camera each).

Each processed camera frame reports, for its PC, the names seen there ("unknown" for
strangers) and how many people are in front of it. Timers only fire an alert after a
situation lasts past its threshold, so someone passing by or sitting briefly never
triggers anything.
"""
import csv
import json
from datetime import datetime, timedelta

import config
from usage import UsageLog


def load_assignments():
    """{"PC-01": ["juan", "maria"], "PC-02": []} — also accepts the old {"PC-01": "juan"} format."""
    if not config.ASSIGNMENTS_FILE.exists():
        return {}
    data = json.loads(config.ASSIGNMENTS_FILE.read_text())
    return {pc: ([v] if isinstance(v, str) and v else [] if isinstance(v, str) else sorted(set(v)))
            for pc, v in data.items()}


def save_assignments(assignments):
    config.ASSIGNMENTS_FILE.write_text(json.dumps(assignments, indent=2))


class Tracker:
    def __init__(self, assignments, watchlist=(), live=True):
        self.set_assignments(assignments, 0.0)
        self.watchlist = set(watchlist)        # people who must not use any PC
        self.live = live
        self.start_wall = datetime.now()
        self.presence = {}       # (pc, name) -> {"first": t, "last": t}   short gap, for alerts
        self.crowd_since = {}    # pc -> t when crowding started
        self.covered_since = {}  # pc -> t when its camera became covered
        self.last_frame = {}     # pc -> t of the last frame from its camera
        self.fired = set()       # (type, pc, name) alerts already raised, so each fires once
        self.spoof_seen = {}     # (pc, name) -> {"first": t, "last": t}  a face shown on a phone/screen
        self.new_alerts = []     # raised since the caller last took them (for incidents)
        config.DATA_DIR.mkdir(exist_ok=True)
        self.usage = UsageLog(config.SESSIONS_LOG, self.stamp, self.assigned_elsewhere)   # everyone, every visit
        self._init_csv(config.ALERTS_LOG, ["time", "type", "pc", "name", "detail"])

    # ---------- helpers ----------
    @staticmethod
    def _init_csv(path, header):
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def stamp(self, t):
        """Wall-clock time for live cameras, offset time for video files."""
        if self.live:
            return (self.start_wall + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S")
        return f"video+{t:.1f}s"

    def _alert(self, t, kind, pc, name, detail):
        key = (kind, pc, name)
        if key in self.fired:
            return
        self.fired.add(key)
        with open(config.ALERTS_LOG, "a", newline="") as f:
            csv.writer(f).writerow([self.stamp(t), kind, pc, name, detail])
        print("ALERT", f"[{self.stamp(t)}] {kind}: {detail}")
        self.usage.note_alert(pc, name, kind)
        self.new_alerts.append({"time": self.stamp(t), "type": kind, "pc": pc, "name": name, "detail": detail})

    def is_assigned(self, pc, name):
        return name in self.assignments.get(pc, ())

    def assigned_elsewhere(self, name):
        """The PCs this person is assigned to (empty if none)."""
        return sorted(self.pcs_of.get(name, ()))

    def online(self, pc, t):
        return t - self.last_frame.get(pc, -1e9) <= config.CAMERA_OFFLINE_SEC

    # ---------- per-frame update for one PC ----------
    def update(self, t, pc, names, people, covered):
        """t: seconds since start. names: names seen at this PC. people: head count. covered: camera blocked."""
        self.last_frame[pc] = t

        if covered:
            since = self.covered_since.setdefault(pc, t)
            if t - since >= config.TAMPER_SEC:
                self._alert(t, "CAMERA_BLOCKED", pc, "-",
                            f"{pc} camera blocked (covered, blank or blinded) for {t - since:.0f}s (possible tampering)")
        else:
            self.covered_since.pop(pc, None)
            self.fired.discard(("CAMERA_BLOCKED", pc, "-"))

        for name in names:
            key = (pc, name)
            p = self.presence.get(key)
            if p is None or t - p["last"] > config.PRESENCE_GAP:
                self.presence[key] = {"first": t, "last": t}
                self.fired = {f for f in self.fired if f[1:] != key}   # new visit: alerts can fire again
            else:
                p["last"] = t
            if self.is_assigned(pc, name):
                self.last_assigned_seen[pc] = t

        if people >= config.CROWD_COUNT:
            since = self.crowd_since.setdefault(pc, t)
            if t - since >= config.CROWD_SEC:
                self._alert(t, "CROWDING", pc, "-", f"{people} people at {pc} for {t - since:.0f}s")
        else:
            self.crowd_since.pop(pc, None)
            self.fired.discard(("CROWDING", pc, "-"))

        self._check_people(t)

    def log_people(self, t, pc, people):
        """Usage log (usage.py): every person this PC camera sees, users and bystanders alike."""
        self.usage.update(t, pc, people)

    def check_liveness(self, t, pc, people):
        """Liveness rules (liveness.py) for one PC camera's people:
        SPOOF: an enrolled student's face shown on a phone/screen for SPOOF_SEC (someone pretending to be them).
        NO_BLINK: the PC's user is a recognized student whose face has not blinked in LIVENESS_SEC (a printed photo?)."""
        for p in people:
            sp = p.get("spoof")
            if sp and sp["of"] not in ("unknown", "unidentified"):
                key = (pc, sp["of"])
                v = self.spoof_seen.get(key)
                if v is None or t - v["last"] > config.PRESENCE_GAP:
                    self.spoof_seen[key] = v = {"first": t, "last": t}
                    self.fired.discard(("SPOOF", pc, sp["of"]))
                v["last"] = t
                if t - v["first"] >= config.SPOOF_SEC:
                    self._alert(t, "SPOOF", pc, sp["of"],
                                f"a face shown on a {sp['device']} at {pc} looks like {sp['of']} (photo or video held up to the camera)")
            lv = p.get("liveness")
            if p.get("user") and lv and lv["state"] == "no_blink" and p["name"] not in ("unknown", "unidentified"):
                self._alert(t, "NO_BLINK", pc, p["name"],
                            f"{p['name']}'s face at {pc} has not blinked in {lv['seconds']}s (a printed photo?)")

    def _check_people(self, t):
        for (pc, name), p in self.presence.items():
            if t - p["last"] > config.PRESENCE_GAP:
                continue   # no longer there
            stayed = t - p["first"]
            if name in self.watchlist:
                if stayed >= config.WATCHLIST_SEC:
                    self._alert(t, "WATCHLIST", pc, name, f"{name} (on watchlist) is using {pc}")
            elif self.is_assigned(pc, name):
                continue
            elif name in self.pcs_of:
                if stayed >= config.WRONG_SEAT_SEC:
                    where = ", ".join(self.assigned_elsewhere(name))
                    self._alert(t, "WRONG_SEAT", pc, name,
                                f"{name} (assigned {where}) is using {pc} for {stayed:.0f}s")
            elif stayed >= config.UNAUTHORIZED_SEC:
                self._alert(t, "UNAUTHORIZED", pc, name, f"{name} is using {pc} for {stayed:.0f}s")

        # Absent: none of a PC's assigned students has been at it for ABSENT_SEC (one alert per PC).
        for pc, names in self.assignments.items():
            if not names:
                continue
            if not self.online(pc, t):
                self.last_assigned_seen[pc] = t   # can't tell while that PC's camera is off
                continue
            gone = t - self.last_assigned_seen.setdefault(pc, t)
            if gone >= config.ABSENT_SEC:
                self._alert(t, "ABSENT", pc, "-",
                            f"none of {pc}'s assigned students ({', '.join(names)}) there for {gone / 60:.1f} min")
            else:
                self.fired.discard(("ABSENT", pc, "-"))

    # ---------- queries ----------
    def status(self, pc, name, t):
        """How a person at a PC is judged right now: ok / watchlist / wrong_seat / unauthorized,
        whether the alert has fired, how long they've been there, and the threshold."""
        p = self.presence.get((pc, name))
        seconds = round(t - p["first"]) if p else 0
        if name in self.watchlist:
            kind, limit = "watchlist", config.WATCHLIST_SEC
        elif self.is_assigned(pc, name):
            return {"state": "ok", "seconds": seconds, "limit": None, "alert": False}
        elif name in self.pcs_of:
            kind, limit = "wrong_seat", config.WRONG_SEAT_SEC
        else:
            kind, limit = "unauthorized", config.UNAUTHORIZED_SEC
        return {"state": kind, "seconds": seconds, "limit": limit,
                "alert": (kind.upper(), pc, name) in self.fired}

    def take_new_alerts(self):
        alerts, self.new_alerts = self.new_alerts, []
        return alerts

    def set_assignments(self, assignments, t):
        """assignments: {"PC-01": ["juan", "maria"], ...}. Changing them restarts the absence timers."""
        self.assignments = {pc: list(names) for pc, names in assignments.items()}
        self.pcs_of = {}                                   # {"juan": {"PC-01"}}
        for pc, names in self.assignments.items():
            for name in names:
                self.pcs_of.setdefault(name, set()).add(pc)
        self.last_assigned_seen = {pc: t for pc in self.assignments}
        if hasattr(self, "fired"):
            self.fired = {f for f in self.fired if f[0] in ("CROWDING", "CAMERA_BLOCKED")}

    def close(self):
        self.usage.close()
