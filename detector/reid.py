"""Short-term person re-identification by appearance (clothing colors + body dimensions).

Why: YOLO's tracker gives a person a NEW track ID whenever they leave the frame, are blocked, or
turn away for a moment. Without re-identification the same person would keep "appending" as
Person #1, #2, #3... and lose their name while their face is not visible.

What is compared (an appearance "signature" per person):
  - upper body (shirt) color: 2-D hue/saturation histogram in HSV (Lesson: Image Data Formats —
    HSV separates color from brightness, so light changes matter less)
  - lower body (pants) color: same, weighted less (often hidden behind the desk)
  - body dimensions: box aspect ratio (height / width) and size relative to the frame
Similarity = weighted mix of histogram similarity (1 - Bhattacharyya distance) and shape similarity.

Faces always win: appearance is only a SHORT-TERM memory (clothes change every day). A recognized
face sets the name; if the face says it is someone else, that is treated as a different person.
"""
import cv2
import numpy as np

import config

H_BINS, S_BINS = 16, 8


def _hist(img, box, top, bottom):
    """HSV hue/saturation histogram of a horizontal band of a person box (fractions of its height)."""
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    ya, yb = int(y1 + top * h), int(y1 + bottom * h)
    xa, xb = int(x1 + 0.2 * w), int(x2 - 0.2 * w)          # central part, avoids background at the sides
    ih, iw = img.shape[:2]
    ya, yb, xa, xb = max(0, ya), min(ih, yb), max(0, xa), min(iw, xb)
    if yb - ya < 8 or xb - xa < 8:
        return None
    hsv = cv2.cvtColor(img[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 30, 30), (180, 255, 255))   # ignore near-gray/dark pixels (no color info)
    if cv2.countNonZero(mask) < 40:
        mask = None                                          # gray/black clothes: use all pixels
    hist = cv2.calcHist([hsv], [0, 1], mask, [H_BINS, S_BINS], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def color_name(img, box):
    """A readable name of the dominant shirt color (for the UI), e.g. 'blue' or 'white'."""
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    (top, bottom), _ = shirt_band(box)
    crop = img[max(0, int(y1 + top * h)):int(y1 + bottom * h), max(0, int(x1 + 0.25 * w)):int(x2 - 0.25 * w)]
    if crop.size == 0:
        return "?"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hue, sat, val = np.median(hsv[:, 0]), np.median(hsv[:, 1]), np.median(hsv[:, 2])
    if val < 60:
        return "black"
    if sat < 40:
        return "white" if val > 170 else "gray"
    for limit, name in [(8, "red"), (20, "orange"), (33, "yellow"), (78, "green"), (100, "cyan"),
                        (130, "blue"), (155, "purple"), (172, "pink"), (181, "red")]:
        if hue < limit:
            return name
    return "?"


FULL_BODY_ASPECT = 1.8   # box height/width above this = standing, whole body visible


def shirt_band(box):
    """Where the shirt is inside a person box (fractions of its height). A PC webcam usually sees a
    seated person from the chest up (wide box): the shirt is the lower part, the face the upper part.
    A standing person (tall box) has the shirt in the upper-middle and the pants below."""
    x1, y1, x2, y2 = box
    full_body = (y2 - y1) / max(1.0, x2 - x1) > FULL_BODY_ASPECT
    return ((0.25, 0.55), (0.60, 0.95)) if full_body else ((0.62, 0.97), None)


def signature(img, box):
    """Appearance signature of one YOLO person box (the face area is left out on purpose)."""
    x1, y1, x2, y2 = box
    ih, iw = img.shape[:2]
    upper, lower = shirt_band(box)
    return {
        "upper": _hist(img, box, *upper),                    # shirt
        "lower": _hist(img, box, *lower) if lower else None, # pants (only when the whole body is visible)
        "aspect": (y2 - y1) / max(1.0, x2 - x1),             # body proportions
        "size": (x2 - x1) * (y2 - y1) / float(iw * ih),      # how big the person is in the frame
        "color": color_name(img, box),
    }


def _hist_sim(a, b):
    if a is None or b is None:
        return None
    return 1.0 - cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)


def clothes_similarity(a, b):
    """0..1 from clothing colors only. Used to decide that a DIFFERENT person took over a track:
    body dimensions are useless there, because everyone sits in the same chair in front of the same camera."""
    parts = [(s, w) for s, w in ((_hist_sim(a["upper"], b["upper"]), 0.75), (_hist_sim(a["lower"], b["lower"]), 0.25))
             if s is not None]
    if not parts:
        return 1.0   # no clothing visible: no evidence of a different person
    return float(sum(s * w for s, w in parts) / sum(w for _, w in parts))


def similarity(a, b):
    """0..1: how alike two signatures are (clothes 75%, body dimensions 25%)."""
    parts, weights = [], []
    for key, wgt in (("upper", 0.55), ("lower", 0.20)):
        s = _hist_sim(a[key], b[key])
        if s is not None:
            parts.append(s); weights.append(wgt)
    aspect = 1.0 - min(1.0, abs(a["aspect"] - b["aspect"]) / max(a["aspect"], b["aspect"]))
    size = 1.0 - min(1.0, abs(a["size"] - b["size"]) / max(a["size"], b["size"], 1e-6))
    parts += [aspect, size]; weights += [0.15, 0.10]
    return float(np.dot(parts, weights) / sum(weights))


def blend(old, new, rate=0.2):
    """Slowly update a stored signature with the latest view (handles small light/pose changes)."""
    out = dict(new)
    for key in ("upper", "lower"):
        if old[key] is not None and new[key] is not None:
            out[key] = (1 - rate) * old[key] + rate * new[key]
        else:
            out[key] = new[key] if new[key] is not None else old[key]
    out["aspect"] = (1 - rate) * old["aspect"] + rate * new["aspect"]
    out["size"] = (1 - rate) * old["size"] + rate * new["size"]
    return out


class Gallery:
    """People recently seen by ONE camera. Maps YOLO track IDs to stable person numbers."""

    def __init__(self):
        self.people = {}     # pid -> {"pid", "name", "score", "at", "sig", "seen", "reid"}
        self.tracks = {}     # YOLO track id -> pid


    def person_for(self, track_id, sig, now, taken):
        """The person this YOLO track belongs to: the same as before, a re-identified recent person
        (by appearance), or a new one. `taken` = pids already matched in this frame."""
        pid = self.tracks.get(track_id) if track_id is not None else None
        if pid in self.people and pid not in taken:
            p = self.people[pid]
            # The tracker links boxes by POSITION, so the next person who sits in the same spot can
            # inherit the previous person's track. If the clothes/body clearly changed, it is someone else.
            if clothes_similarity(p["sig"], sig) >= config.REID_SPLIT_SIM:
                p["sig"] = blend(p["sig"], sig)   # "reid" stays until a face confirms the name again
                if p["name"] is None:             # face never seen: maybe a named person whose box was lost
                    named = self._takeover(p, sig, now, taken)
                    if named is not None:
                        self.tracks[track_id] = named["pid"]
                        return named
                return p
            self.tracks.pop(track_id, None)
        best, best_s = None, 0.0
        for p in self.people.values():
            if p["pid"] in taken or now - p["seen"] > config.REID_WINDOW_SEC:
                continue
            s = similarity(p["sig"], sig)
            if s > best_s:
                best, best_s = p, s
        if best is not None and best_s >= config.REID_MIN_SIM:
            best["sig"], best["reid"], best["reid_sim"] = blend(best["sig"], sig), True, round(best_s, 2)
            if track_id is not None:
                self.tracks[track_id] = best["pid"]
            return best
        return self.new_person(track_id, sig, now)

    def _takeover(self, p, sig, now, taken):
        """YOLO sometimes puts a second, short-lived box on the same person (e.g. around the head and a
        raised hand). That box gets its own nameless identity, and if the main box then disappears the
        person is left nameless. So a person whose face was NEVER seen takes over a named person who has
        been out of view for REID_TAKEOVER_SEC (so two people in view never swap names) and whose clothes
        and body match. A person whose face was seen but not recognized (a stranger) is never renamed."""
        best, best_s = None, 0.0
        for q in self.people.values():
            if (q is p or q["pid"] in taken or q["name"] in (None, "unknown")
                    or not config.REID_TAKEOVER_SEC <= now - q["seen"] <= config.REID_WINDOW_SEC):
                continue
            s = similarity(q["sig"], sig)
            if s > best_s:
                best, best_s = q, s
        if best is None or best_s < config.REID_MIN_SIM:
            return None
        del self.people[p["pid"]]
        best["sig"], best["reid"], best["reid_sim"] = blend(best["sig"], sig), True, round(best_s, 2)
        return best

    def new_person(self, track_id, sig, now, name=None, score=0.0):
        pid = 1
        while pid in self.people:    # reuse the lowest free number, so numbers don't keep growing
            pid += 1
        p = {"pid": pid, "name": name, "score": score, "at": 0.0 if name is None else now,
             "sig": sig, "seen": now, "reid": False}
        self.people[pid] = p
        if track_id is not None:
            self.tracks[track_id] = pid
        return p

    def forget_old(self, now):
        """Named people are remembered for REID_WINDOW_SEC. People never identified (unknown / face never
        seen) only for UNNAMED_FORGET_SEC: when the camera is moved or covered their appearance changes
        anyway, and keeping them would only pile up "new" people."""
        def keep(p):
            limit = config.REID_WINDOW_SEC if p["name"] not in (None, "unknown") else config.UNNAMED_FORGET_SEC
            return now - p["seen"] <= limit
        self.people = {k: v for k, v in self.people.items() if keep(v)}
        self.tracks = {t: pid for t, pid in self.tracks.items() if pid in self.people}
