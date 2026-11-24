"""Usage log: one row per person per visit to a PC camera, for EVERYONE the camera sees.

Each row says what kind of visit it was, so nothing is lost but the real PC usage stays easy to find:
  - role "user": the person in the PC's seat (the biggest / closest YOLO26 box) most of the time.
    A named user is counted PC usage; "unknown" = a face that is not enrolled, "unidentified" = face never seen.
  - role "bystander": someone standing behind or next to the user most of the time.
  - role "spoof": a face shown on a phone/screen (liveness.py), named after the student it looks like.

When does a visit end? When the person has not been seen for SESSION_GAP seconds (2 min). Leaving for
30 minutes and coming back therefore gives a NEW row. Shorter absences (stood up, bent down, the camera
missed them) stay in the same row and are counted as breaks, with the time away.

A visit is followed by the person's re-ID number (uid), so it keeps going while their face is hidden, and
by their name, so a person who comes back as a "new" person within SESSION_GAP continues their row. A
person who was "unidentified" at first takes their name as soon as their face is recognized.
"""
import csv
from collections import Counter
from itertools import count

import config

FIELDS = ["start", "end", "name", "pc", "role", "duration_sec", "active_sec", "breaks", "away_sec",
          "assigned_here", "assigned_to", "identified_by", "best_score", "frames", "alerts"]
UNNAMED = ("unknown", "unidentified")
MIN_LOG_SEC = 3          # visits shorter than this are glitches (someone walking past), not logged


def migrate(path):
    """Old usage logs had name,pc,start,end,minutes,assigned_here: convert them to the new columns."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == FIELDS:
            return
        rows = list(reader)
    out = []
    for r in rows:
        sec = round(float(r.get("minutes") or 0) * 60) if "minutes" in r else int(float(r.get("duration_sec") or 0))
        out.append({k: "" for k in FIELDS} | {
            "start": r.get("start", ""), "end": r.get("end", ""), "name": r.get("name", ""), "pc": r.get("pc", ""),
            "role": r.get("role") or "user", "duration_sec": sec, "active_sec": r.get("active_sec") or sec,
            "breaks": r.get("breaks") or 0, "away_sec": r.get("away_sec") or 0,
            "assigned_here": r.get("assigned_here", "")})
    write_all(path, out)


def write_all(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_all(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class UsageLog:
    def __init__(self, path, stamp, assigned_to):
        """stamp(t) -> wall-clock text; assigned_to(name) -> list of PCs that person is assigned to."""
        self.path, self.stamp, self.assigned_to = path, stamp, assigned_to
        self.open = {}            # visit id -> visit
        self.ids = count(1)
        migrate(path)
        if not path.exists():
            write_all(path, [])

    def update(self, t, pc, people):
        """people: from detection.analyze() for this PC camera (with "user", "uid", "name", "spoof", "reid")."""
        here = set()
        for p in people:
            sp = p.get("spoof")
            kind = "spoof" if sp else "person"
            name = sp["of"] if sp else p["name"]
            v = self._find(t, pc, p.get("uid"), name, kind, here)
            if v is None:
                v = {"id": next(self.ids), "pc": pc, "uid": p.get("uid"), "name": name, "kind": kind,
                     "start": t, "last": t, "frames": 0, "user_frames": 0, "breaks": 0, "away": 0.0,
                     "how": Counter(), "best": 0.0, "alerts": Counter()}
                self.open[v["id"]] = v
            here.add(v["id"])
            gap = t - v["last"]
            if gap > config.PRESENCE_GAP:          # was gone for a while, but not long enough to end the visit
                v["breaks"] += 1
                v["away"] += gap
            if v["name"] in UNNAMED and name not in UNNAMED:
                v["name"] = name                   # face recognized now: the whole visit is theirs
            elif v["name"] == "unidentified" and name == "unknown":
                v["name"] = name
            v["uid"] = p.get("uid", v["uid"])
            v["last"] = t
            v["frames"] += 1
            v["user_frames"] += 1 if p.get("user") else 0
            v["how"]["phone/screen" if sp else "clothes" if p.get("reid") else "face" if p.get("face_box")
                     else "tracking"] += 1
            score = (p.get("checked") or {}).get("score") if sp else p.get("score")
            v["best"] = max(v["best"], float(score or 0))
        for vid, v in list(self.open.items()):
            if t - v["last"] > config.SESSION_GAP:
                self._close(v)
                del self.open[vid]

    def _find(self, t, pc, uid, name, kind, taken):
        """The open visit this person continues: same name (if known) or same re-ID person, recent enough."""
        live = [v for v in self.open.values() if v["pc"] == pc and v["kind"] == kind and v["id"] not in taken
                and t - v["last"] <= config.SESSION_GAP]
        if name not in UNNAMED:
            v = next((v for v in live if v["name"] == name), None)
            if v is not None:
                return v
        v = next((v for v in live if uid is not None and v["uid"] == uid
                  and (v["name"] in UNNAMED or name in UNNAMED or v["name"] == name)), None)
        return v

    def note_alert(self, pc, name, kind):
        """An alert raised for this person at this PC is written on their visit."""
        for v in self.open.values():
            if v["pc"] == pc and v["name"] == name:
                v["alerts"][kind] += 1

    def row(self, v, live=False):
        dur = v["last"] - v["start"]
        role = "spoof" if v["kind"] == "spoof" else "user" if v["user_frames"] * 2 >= v["frames"] else "bystander"
        assigned = self.assigned_to(v["name"]) if v["name"] not in UNNAMED else []
        total = sum(v["how"].values()) or 1
        return {
            "start": self.stamp(v["start"]), "end": "" if live else self.stamp(v["last"]),
            "name": v["name"], "pc": v["pc"], "role": role,
            "duration_sec": round(dur), "active_sec": round(max(0.0, dur - v["away"])),
            "breaks": v["breaks"], "away_sec": round(v["away"]),
            "assigned_here": v["pc"] in assigned, "assigned_to": " ".join(assigned),
            "identified_by": " ".join(f"{k}:{round(100 * n / total)}%" for k, n in v["how"].most_common()),
            "best_score": round(v["best"], 2), "frames": v["frames"],
            "alerts": " ".join(f"{k}x{n}" if n > 1 else k for k, n in v["alerts"].items()),
            **({"live": True, "seen_ago": None} if live else {}),
        }

    def _close(self, v):
        if v["last"] - v["start"] < MIN_LOG_SEC:
            return
        if not self.path.exists():         # (deleted by a reset)
            write_all(self.path, [])
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore").writerow(self.row(v))

    def active(self, t):
        rows = []
        for v in self.open.values():
            if v["last"] - v["start"] < MIN_LOG_SEC:
                continue
            r = self.row(v, live=True)
            r["seen_ago"] = round(t - v["last"])
            rows.append(r)
        return rows

    def close(self):
        for v in self.open.values():
            self._close(v)
        self.open.clear()
