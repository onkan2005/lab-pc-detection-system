"""Liveness: is the recognized face a real person, or a photo / phone held up to the camera?

ArcFace only compares faces, so a printed photo or a phone showing a student's face matches that
student just as well as the student themself. Two checks catch this, both with pretrained models:

1. Face on a phone / screen (YOLO26): YOLO26 also looks for COCO "cell phone", "tv" and "laptop".
   A face inside a phone's box, or a person who is entirely inside a screen's box, is being SHOWN to
   the camera, not sitting in front of it.
2. Blinking (InsightFace 2d106det, 106 face landmarks): people blink every few seconds; a printed photo
   never does. Eye openness = gap between the eyelids / distance from the eyes to the mouth. Dividing by
   the eye-to-mouth distance makes it immune to tilting a photo back and forth (both shrink together).
   A blink = openness drops below BLINK_RATIO of that person's usual openness. A recognized face that is
   visible for LIVENESS_SEC without a single blink is flagged.

Limits (honest): a VIDEO of the student played on a phone does blink, so it is only caught when YOLO26
sees the phone. Blinks are only measured on PC cameras; CCTV faces are too small for eyelid landmarks.
"""
import numpy as np

import config

# 2d106det landmark numbers (checked on an enrolled photo): eye corners, then upper/lower eyelid pairs
EYES = [((35, 39), [(41, 36), (40, 33), (42, 37)]),
        ((89, 93), [(95, 90), (94, 87), (96, 91)])]
HISTORY = 40          # openness samples kept per person (the "usual" openness is their median)
MIN_SAMPLES = 6       # samples needed before a blink can be judged


def openness(lmk, kps):
    """Eyelid gap / eyes-to-mouth distance (about 0.10-0.20 open, much lower when closed)."""
    gap = np.mean([np.linalg.norm(lmk[lo] - lmk[up]) for _, pairs in EYES for up, lo in pairs])
    eyes, mouth = (kps[0] + kps[1]) / 2, (kps[3] + kps[4]) / 2
    return float(gap / max(1.0, np.linalg.norm(mouth - eyes)))


def blink_step(state, value, now):
    """Adds one openness sample to a person's liveness state. Returns True if this sample is a blink."""
    hist = state.setdefault("hist", [])
    usual = float(np.median(hist)) if len(hist) >= MIN_SAMPLES else None
    closed = usual is not None and value < config.BLINK_RATIO * usual
    blink = closed and not state.get("closed", False)    # count the closing, not every closed frame
    state["closed"] = closed
    if not closed:
        hist.append(value)                                # usual openness only from open-eye samples
        del hist[:-HISTORY]
    if blink:
        state["blinks"] = state.get("blinks", 0) + 1
        state["last_blink"] = now
    return blink


def update(person, face_app, img, face, now):
    """Liveness for one person whose face is visible in this frame (PC cameras).
    Keeps its state in the person's gallery entry, so it follows them like their name does."""
    st = person.setdefault("live", {"since": now, "visible": 0.0, "blinks": 0, "last": now})
    st["visible"] += min(1.0, now - st["last"])           # seconds the face was actually visible
    st["last"] = now
    model = face_app.models.get("landmark_2d_106")
    if model is None:
        return st
    lmk = model.get(img, face)
    value = openness(lmk, np.asarray(face.kps))
    st["openness"] = round(value, 3)
    blink_step(st, value, now)
    return st


def state_of(person):
    """live / checking / no_blink, for the labels and the NO_BLINK rule."""
    st = person.get("live")
    if not st:
        return None
    if st.get("blinks"):
        return {"state": "live", "blinks": st["blinks"], "seconds": round(st["visible"]), "openness": st.get("openness")}
    return {"state": "no_blink" if st["visible"] >= config.LIVENESS_SEC else "checking", "openness": st.get("openness"),
            "blinks": 0, "seconds": round(st["visible"]), "limit": config.LIVENESS_SEC}


def _inside(box, outer, margin=0.1):
    """Share of `box` that lies inside `outer` (grown by `margin` of its size on each side)."""
    ox1, oy1, ox2, oy2 = outer
    mx, my = margin * (ox2 - ox1), margin * (oy2 - oy1)
    ox1, oy1, ox2, oy2 = ox1 - mx, oy1 - my, ox2 + mx, oy2 + my
    x1, y1, x2, y2 = box
    iw, ih = max(0.0, min(x2, ox2) - max(x1, ox1)), max(0.0, min(y2, oy2) - max(y1, oy1))
    return iw * ih / max(1e-6, (x2 - x1) * (y2 - y1))


def on_device(face_box, person_box, devices):
    """The device a face is shown on, or None. devices: [(kind, box), ...] from YOLO26.
    Phone: the face is inside the phone. Screen/laptop: the whole PERSON is inside the screen (a real
    person standing in front of a monitor sticks out of it; a person in a video call does not)."""
    for kind, box in devices:
        if kind == "phone" and _inside(face_box, box) >= 0.8:
            return kind
        if kind != "phone" and _inside(face_box, box) >= 0.8 and _inside(person_box, box, 0.02) >= 0.9:
            return kind
    return None
