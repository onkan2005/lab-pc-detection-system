"""Labels and colors for each YOLO-tracked person, plus drawing them on a frame
(for evidence snapshots and the command-line version).

The web page draws the same labels itself on top of the live video, using describe().
"""
import time

import cv2

from detection import UNIDENTIFIED
from labels import StableNumbers

COLORS = {"ok": "#1f9d55", "checking": "#f08c00", "alert": "#e03131", "bystander": "#868e96"}
_numbers = StableNumbers()   # "Unknown person N" numbers, per PC camera


def describe(pc, result, tracker, t):
    """For each person: what to show. Returns the people with "label", "color", "kind" and rule status added."""
    # People are shown by who they ARE: an enrolled student by name, anyone else as "Unknown person N".
    # N stays with that person while they are in view and is reused after they leave (labels.py), so it
    # never grows past the number of people around and does not jump when people move.
    unnamed = [p for p in result["people"] if p["name"] in (UNIDENTIFIED, "unknown")]
    slot = _numbers.assign(pc, unnamed, t)
    out = []
    for p in result["people"]:
        name = p["name"]
        n = f" {slot[p['id']]}" if p["id"] in slot else ""
        who = f"Unknown person{n}" if name in ("unknown", UNIDENTIFIED) else name
        if p.get("reid") and name not in (UNIDENTIFIED, "unknown"):
            who += f" (by {p.get('clothes', '')} clothes)"   # face hidden: kept by appearance re-ID
        st = tracker.status(pc, name, t) if name != UNIDENTIFIED else \
            {"state": "unidentified", "seconds": 0, "limit": None, "alert": False}
        sp, lv, hint = p.get("spoof"), p.get("liveness"), p.get("hint")
        if sp:
            of = sp["of"] if sp["of"] not in ("unknown", UNIDENTIFIED) else None
            fired = of and ("SPOOF", pc, of) in tracker.fired
            label = (f"{'ALERT: ' if fired else ''}Face on a {sp['device']}" + (f" - looks like {of}, not counted" if of else " - ignored"))
            kind = "alert" if of else "bystander"
        elif name == UNIDENTIFIED:
            label, kind = f"Person{n} - looking for face", "bystander"
            if hint:
                label = f"Person{n} - maybe {hint['name']}? (clothes match, seen on {hint['camera']} {hint['ago']}s ago)"
        elif not p["user"] and name not in tracker.watchlist:
            label, kind = f"{who} (bystander)", "bystander"
        elif p["user"] and lv and lv["state"] == "no_blink":
            fired = ("NO_BLINK", pc, name) in tracker.fired
            label, kind = f"{'ALERT: ' if fired else ''}{who} - no blink in {lv['seconds']}s (photo?)", "alert" if fired else "checking"
        elif st["state"] == "ok":
            label, kind = f"{who} - assigned to {pc}", "ok"
        elif st["state"] == "watchlist":
            label, kind = f"WATCHLIST: {who}", "alert"
        else:
            if st["state"] == "wrong_seat":
                problem = f"assigned to {', '.join(tracker.assigned_elsewhere(name))}, not {pc}"
            else:
                problem = f"not assigned to {pc}"
            if st["alert"]:
                label, kind = f"ALERT: {who} - {problem}", "alert"
            else:
                label, kind = f"{who} - {problem} ({st['seconds']}s/{st['limit']}s)", "checking"
        out.append({**p, **st, "label": label, "color": COLORS[kind], "kind": kind, "display": who})
    return out


def _bgr(hex_color):
    h = hex_color.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def privacy_region(p, w, h):
    """The area to hide for one person: the face box (with a margin), or the head area of the
    YOLO person box when no face is visible."""
    if p.get("face_box"):
        x1, y1, x2, y2 = p["face_box"]
        mx, my = 0.25 * (x2 - x1), 0.25 * (y2 - y1)
        x1, y1, x2, y2 = x1 - mx, y1 - my, x2 + mx, y2 + my
    else:
        bx1, by1, bx2, by2 = p["box"]
        bw, bh = bx2 - bx1, by2 - by1
        full_body = bh / max(1, bw) > 1.8
        x1, x2 = bx1 + 0.2 * bw, bx2 - 0.2 * bw
        y1, y2 = by1, by1 + (0.25 if full_body else 0.6) * bh
    return max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))


def blur_faces(frame, people, block=18):
    """Privacy: pixelate every person's face/head in place. Pixelation (shrink, then enlarge without
    smoothing) throws the detail away, unlike a light blur."""
    h, w = frame.shape[:2]
    for p in people:
        x1, y1, x2, y2 = privacy_region(p, w, h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = frame[y1:y2, x1:x2]
        small = cv2.resize(roi, (max(1, (x2 - x1) // block), max(1, (y2 - y1) // block)), interpolation=cv2.INTER_AREA)
        frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return frame


def draw_boxes(frame, people):
    """Clean version for the camera wall thumbnails: only thin YOLO person boxes, no text."""
    for p in people:
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), _bgr(p["color"]), 2, cv2.LINE_AA)
    return frame


def draw(frame, pc, people, covered=False):
    """Draws YOLO person boxes (+ face boxes) with labels and a camera header bar on the frame (in place)."""
    h, w = frame.shape[:2]
    if covered:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 90), -1)
        frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        text = "CAMERA BLOCKED"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.putText(frame, text, ((w - tw) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    for p in people:
        color = _bgr(p["color"])
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4 if p["kind"] == "alert" else 2)
        if p.get("face_box"):
            fx1, fy1, fx2, fy2 = p["face_box"]
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 1)
        (tw, th), _ = cv2.getTextSize(p["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1, th + 36)
        cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 8, ty), color, -1)
        cv2.putText(frame, p["label"], (x1 + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.circle(frame, (14, 14), 6, (0, 0, 230), -1)
    cv2.putText(frame, f"REC  {pc} camera  {time.strftime('%Y-%m-%d %H:%M:%S')}  YOLO26 people: {len(people)}",
                (28, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
