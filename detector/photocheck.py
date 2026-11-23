"""Check a photo: run one photo through the same steps as the live system and show the numbers.

Steps (each timed): preprocessing (course lessons) -> YOLO26 people -> SCRFD faces -> ArcFace embedding
-> matching against every enrolled photo (cosine similarity). For each face it also measures what
decides whether the live system can recognize it: face size, sharpness, head pose (from the five face
landmarks), and how far the best match is above the threshold and above the next person.
"""
import base64
import time

import cv2
import numpy as np
from insightface.app.common import Face

import config
import preprocess as pp
from detection import sharpness

WORK_WIDTH = 1280          # big photos are shrunk to this for the measurements (faces stay large enough)


def head_pose(kps):
    """Rough head pose from the 5 SCRFD landmarks (left eye, right eye, nose, mouth left, mouth right).
    yaw: nose offset from the middle of the eyes, in eye-distances (0 = facing the camera).
    pitch: where the nose sits between the eyes and the mouth (about 0.5 when looking straight)."""
    le, re_, nose, ml, mr = kps
    eye_mid, mouth_mid = (le + re_) / 2, (ml + mr) / 2
    eye_dist = max(1.0, float(np.linalg.norm(re_ - le)))
    yaw = float((nose[0] - eye_mid[0]) / eye_dist)
    pitch = float((nose[1] - eye_mid[1]) / max(1.0, mouth_mid[1] - eye_mid[1]))
    side = "front" if abs(yaw) < 0.15 else ("turned to their left" if yaw > 0 else "turned to their right")
    vert = "looking down" if pitch > 0.78 else "looking up" if pitch < 0.35 else ""
    return {"yaw": round(yaw, 2), "pitch": round(pitch, 2),
            "label": side if not vert else (vert if side == "front" else f"{side}, {vert}")}


def detect(face_app, img):
    """SCRFD on the whole photo; if nothing is found (a face filling the whole selfie), add a border and retry."""
    size = (config.FACE_DET_SIZE, config.FACE_DET_SIZE)
    bboxes, kpss = face_app.det_model.detect(img, input_size=size, max_num=0)
    if len(bboxes):
        return bboxes, kpss, False
    pad = max(img.shape[:2]) // 2
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    bboxes, kpss = face_app.det_model.detect(padded, input_size=size, max_num=0)
    if len(bboxes):
        bboxes[:, :4] -= pad
        kpss = kpss - pad
    return bboxes, kpss, True


def check_photo(img, face_app, yolo, recognizer):
    timings, t = {}, time.perf_counter

    # 0. image + preprocessing (Lessons 1-3)
    t0 = t()
    h0, w0 = img.shape[:2]
    if w0 > WORK_WIDTH:
        img = cv2.resize(img, (WORK_WIDTH, round(h0 * WORK_WIDTH / w0)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    enhanced, prep = pp.enhance(img)
    timings["preprocess"] = (t() - t0) * 1000

    # 1. YOLO26: people in the photo
    t0 = t()
    det = yolo.predict(enhanced, classes=[0], conf=config.YOLO_CONF, imgsz=640, verbose=False)[0]
    persons = [{"box": [round(v) for v in b], "conf": round(c, 2)}
               for b, c in zip(det.boxes.xyxy.tolist(), det.boxes.conf.tolist())]
    timings["yolo"] = (t() - t0) * 1000

    # 2. SCRFD: faces (+ 5 landmarks each)
    t0 = t()
    bboxes, kpss, padded = detect(face_app, enhanced)
    timings["face_detect"] = (t() - t0) * 1000

    # 3. ArcFace: one 512-number embedding per face   4. matching: cosine similarity with every enrolled photo
    names = [str(n) for n in recognizer.names]
    emb = np.asarray(recognizer.embeddings)
    faces, t_arc, t_match = [], 0.0, 0.0
    order = sorted(range(len(bboxes)), key=lambda k: bboxes[k][0])
    for n, k in enumerate(order, 1):
        b, kps = bboxes[k], kpss[k]
        face = Face(bbox=b[:4], kps=kps, det_score=b[4])
        t0 = t()
        face_app.models["recognition"].get(enhanced, face)
        t_arc += (t() - t0) * 1000
        t0 = t()
        sims = emb @ face.normed_embedding if len(names) else np.zeros(0)
        per_person = {}
        for nm, s in zip(names, sims):
            per_person[nm] = max(per_person.get(nm, -1.0), float(s))
        ranked = sorted(per_person.items(), key=lambda kv: -kv[1])
        t_match += (t() - t0) * 1000
        best_name, best = ranked[0] if ranked else ("unknown", 0.0)
        second = ranked[1][1] if len(ranked) > 1 else None
        name = best_name if best >= config.MATCH_THRESHOLD else "unknown"
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        fw = round(x2 - x1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        inside = next((p for p in persons if p["box"][0] <= cx <= p["box"][2] and p["box"][1] <= cy <= p["box"][3]), None)
        sharp = sharpness(enhanced, b[:4])
        pose = head_pose(kps)
        faces.append({
            "n": n, "name": name, "score": round(best, 3), "best_name": best_name,
            "second": round(second, 3) if second is not None else None,
            "margin": round(best - second, 3) if second is not None else None,
            "per_person": {k2: round(v, 3) for k2, v in ranked},
            "photos_compared": len(names),
            "box": [round(x1), round(y1), round(x2), round(y2)], "width": fw,
            "det_score": round(float(b[4]), 2), "sharpness": round(sharp), "pose": pose,
            "in_person": inside,
            "checks": {   # what the live system needs (see config.py)
                "pc_size": fw >= config.MIN_FACE_PX, "cctv_size": fw >= config.CCTV_MIN_FACE_PX,
                "sharp": sharp >= config.FACE_SHARPNESS_MIN, "front": pose["label"] == "front",
                "in_person": inside is not None,
            },
            "embedding_preview": [round(float(v), 3) for v in face.normed_embedding[:8]],
            "same_photo": best > 0.97,   # almost identical to an enrolled photo: probably that very photo
        })
    timings["arcface"] = t_arc
    timings["match"] = t_match

    # annotated photo: shrink first, then draw, so the labels are readable on any photo size
    k = min(1.0, 900 / max(h, w))
    out = cv2.resize(enhanced, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else enhanced.copy()

    def tag(text, x, y, color):
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(y, tht + 8)
        cv2.rectangle(out, (x, y - tht - 8), (x + tw + 8, y), color, -1)
        cv2.putText(out, text, (x + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for p in persons:
        x1, y1, x2, y2 = (round(v * k) for v in p["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (160, 160, 160), 2)
        tag(f"person {p['conf']:.2f}", x1, y1 + 22, (120, 120, 120))
    for f in faces:
        x1, y1, x2, y2 = (round(v * k) for v in f["box"])
        color = (85, 157, 31) if f["name"] != "unknown" else (49, 49, 224)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        for x, y in kpss[order[f["n"] - 1]]:
            cv2.circle(out, (int(x * k), int(y * k)), 3, (0, 215, 255), -1)
        tag(f"#{f['n']} {f['name']} {f['score']:.2f}", x1, y1, color)
    data = base64.b64encode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()

    return {
        "image": f"data:image/jpeg;base64,{data}",
        "photo": {"width": w0, "height": h0, "analyzed": f"{w}x{h}", "padded_retry": padded,
                  "brightness": prep["before"]["brightness"], "contrast": prep["before"]["contrast"],
                  "corrected": prep["steps"], "after": [prep["after"]["brightness"], prep["after"]["contrast"]],
                  "hist": pp.histogram(img, 64)},
        "persons": persons, "faces": faces,
        "timings": {k: round(v, 1) for k, v in timings.items()},
        "threshold": config.MATCH_THRESHOLD, "enrolled_people": len(set(names)), "enrolled_photos": len(names),
        "limits": {"pc_face": config.MIN_FACE_PX, "cctv_face": config.CCTV_MIN_FACE_PX,
                   "sharpness": config.FACE_SHARPNESS_MIN},
    }
