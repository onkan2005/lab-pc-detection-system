"""Detection for one camera, built around YOLO26.

A PC camera belongs to a PC, so the person sitting in front of it is that PC's user. A CCTV camera
(CCTV_PROFILE) watches the whole room: same steps, bigger frames, smaller faces, no "user".
Per frame:
  0. Preprocessing (preprocess.py, earlier lessons): brightness/contrast are measured; a dark or
     flat frame is enhanced (min-max normalization by default); glare is found with the digital negative.
  1. YOLO26 detects every person and ByteTrack gives each one a track ID (#1, #2, ...)
     that stays with them from frame to frame.
  2. Tamper check: blinded by light (glare), or almost no detail while YOLO sees nobody = blocked.
  3. The largest YOLO person box (the one closest to the camera) is the PC's user;
     other people are bystanders.
  4. Inside each YOLO person box, SCRFD looks for the face; ArcFace turns it into an
     embedding and it is matched with the enrolled students.
  5. The name is remembered per track ID, so a person who turns away keeps their name;
     it is re-checked every RECOGNIZE_EVERY seconds.
"""
import math
import time

import cv2
from insightface.app.common import Face
from ultralytics import YOLO

import config
from preprocess import detail, enhance, stats
from reid import Gallery, signature

UNIDENTIFIED = "unidentified"   # a person YOLO sees whose face has not been recognized yet


# Detection sizes per camera type. A PC webcam sees one person up close; a CCTV room camera sees
# many people far away, so it keeps a bigger image and looks for smaller faces.
PC_PROFILE = {"width": config.DETECT_WIDTH, "yolo": config.YOLO_IMGSZ,
              "face": config.LIVE_FACE_DET_SIZE, "min_face": config.MIN_FACE_PX}
CCTV_PROFILE = {"width": config.CCTV_DETECT_WIDTH, "yolo": config.CCTV_YOLO_IMGSZ,
                # the face detector runs on each person's crop, enlarged to 640 px so small faces are found
                "face": 640, "min_face": config.CCTV_MIN_FACE_PX,
                "tracker": str(config.BASE_DIR / "tracker_cctv.yaml")}


def new_camera(profile=PC_PROFILE):
    """State for one camera: its own YOLO26 model (so its tracker only follows this camera's
    people), the gallery of people recently seen by it (re-identification by appearance), and
    its detection sizes (PC webcam or CCTV)."""
    return {"yolo": YOLO(config.YOLO_MODEL), "gallery": Gallery(), "profile": profile}


def shrink(frame, width=config.DETECT_WIDTH):
    """Resize to `width` for speed. Returns the small frame and the scale back to the original."""
    h, w = frame.shape[:2]
    if w <= width:
        return frame, 1.0
    s = width / w
    return cv2.resize(frame, (width, round(h * s))), 1 / s


def sharpness(img, bbox):
    """How sharp a face is: variance of the Laplacian (edge filter) of the face, resized to 112x112.
    Sharp faces give large values (many strong edges); blurred / out-of-focus faces give small ones."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return 0.0
    gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (112, 112))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def find_face(img, person_box, face_app, det_size=config.LIVE_FACE_DET_SIZE, min_face=config.MIN_FACE_PX):
    """Looks for a face only inside a YOLO person box. Returns an insightface Face (full-image
    coordinates) or None."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in person_box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < min_face or y2 - y1 < min_face:
        return None
    crop = img[y1:y2, x1:x2]
    size = (det_size, det_size)
    bboxes, kpss = face_app.det_model.detect(crop, input_size=size, max_num=0)
    if len(bboxes) == 0:
        return None
    i = max(range(len(bboxes)), key=lambda k: (bboxes[k][2] - bboxes[k][0]) * (bboxes[k][3] - bboxes[k][1]))
    b = bboxes[i]
    if b[2] - b[0] < min_face:
        return None
    offset = [x1, y1, x1, y1]
    kps = kpss[i] + [x1, y1] if kpss is not None else None
    return Face(bbox=b[:4] + offset, kps=kps, det_score=b[4])


def analyze(frame, cam, face_app, recognizer, now=None, timings=None):
    """Returns {"covered", "persons", "people": [{"id", "box", "face_box", "name", "score", "user"}], "user"}.

    cam: the state from new_camera() for this PC. Boxes are in the coordinates of the frame passed in.
    now: clock in seconds (default: time.monotonic(); a video test passes the video's own time).
    timings: optional dict; milliseconds spent in each step are added to it (preprocess, yolo, reid,
    face_detect, arcface) plus "arcface_runs", so the video test can show where the time goes.
    """
    t_ms = timings if timings is not None else {}
    clock = time.perf_counter

    def spent(step, since):
        t_ms[step] = t_ms.get(step, 0.0) + (clock() - since) * 1000

    prof = cam.get("profile", PC_PROFILE)
    t0 = clock()
    small, back = shrink(frame, prof["width"])
    # 0. Preprocessing from the earlier lessons: measure brightness/contrast, fix dark or flat frames
    if config.PREPROCESS:
        small, prep = enhance(small)
    else:
        prep = {"before": stats(small), "after": stats(small), "steps": [], "glare": False,
                "detail": round(detail(small), 1), "format": f"{small.shape[1]}x{small.shape[0]} {small.dtype} BGR"}
    spent("preprocess", t0)
    if prep["glare"]:
        return {"covered": True, "covered_reason": "blinded by a bright light", "persons": 0,
                "people": [], "user": None, "prep": prep}
    now = time.monotonic() if now is None else now

    # 1-2. YOLO26 person detection + tracker IDs
    # YOLO input size: the profile's size, but never much bigger than the camera image itself. Blowing a
    # low-resolution CCTV frame up only makes it blurry: on a 384-px CCTV video YOLO26 found the person in
    # 126/145 frames at 480 px, 119 at 640, 95 at 960 (measured with evaluate_video.py test clips).
    t0 = clock()
    imgsz = min(prof["yolo"], max(320, math.ceil(small.shape[1] * 1.25 / 32) * 32))
    det = cam["yolo"].track(small, persist=True, tracker=prof.get("tracker", "bytetrack.yaml"), classes=[0],
                            conf=prof.get("conf", config.YOLO_CONF), imgsz=imgsz, verbose=False)[0]
    boxes = det.boxes.xyxy.tolist()
    confs = det.boxes.conf.tolist()
    spent("yolo", t0)
    prep["yolo_imgsz"] = imgsz
    # Tamper check: almost no detail AND YOLO sees nobody. (A dim room still has detail and people,
    # so it is not mistaken for a covered lens.)
    if not boxes and prep["detail"] < config.TAMPER_STD:
        return {"covered": True, "covered_reason": "covered or blank", "persons": 0,
                "people": [], "user": None, "prep": prep}
    ids = det.boxes.id.int().tolist() if det.boxes.id is not None else [None] * len(boxes)

    gallery, people, taken = cam["gallery"], [], set()
    # biggest boxes first, so the PC's user gets first pick when matching appearances
    order = sorted(range(len(boxes)), key=lambda k: -(boxes[k][2] - boxes[k][0]) * (boxes[k][3] - boxes[k][1]))
    for k in order:
        box, tid = boxes[k], ids[k]
        # 3b. re-identification: same track, or a recent person with the same clothes/body -> same person
        t0 = clock()
        sig = signature(small, box)
        person = gallery.person_for(tid, sig, now, taken)
        spent("reid", t0)
        # 4. face inside this person's box (cheap) -> recognize only when needed (ArcFace is the slow part)
        t0 = clock()
        face = find_face(small, box, face_app, prof["face"], prof["min_face"])
        spent("face_detect", t0)
        checked = None
        if face is not None and (person["name"] in (None, "unknown") or now - person["at"] >= config.RECOGNIZE_EVERY):
            t0 = clock()
            face_app.models["recognition"].get(small, face)
            name, score = recognizer.identify(face.normed_embedding)
            spent("arcface", t0)
            t_ms["arcface_runs"] = t_ms.get("arcface_runs", 0) + 1
            name = str(name)
            checked = {"name": name, "score": round(float(score), 3)}   # this frame's ArcFace answer
            known = person["name"] not in (None, "unknown")
            clear = sharpness(small, face.bbox) >= config.FACE_SHARPNESS_MIN
            if known and name == "unknown" and not clear:
                person["at"] = now          # unclear face (blur, angle): keep who the clothes/track say it is
            elif known and name != person["name"] and (name != "unknown" or clear):
                # a clear face says this is someone else (e.g. same uniform): split into a new person
                person = gallery.new_person(tid, sig, now, name, float(score))
            else:
                person.update(name=name, score=float(score), at=now, reid=False)
        person["seen"] = now
        taken.add(person["pid"])
        people.append({
            "id": person["pid"], "track": tid, "name": person["name"] or UNIDENTIFIED,
            "score": round(person["score"], 2), "user": False,
            "reid": bool(person.get("reid")),   # name from clothes/body re-ID, not yet confirmed by a face
            "reid_sim": person.get("reid_sim"),
            "clothes": sig["color"],
            "conf": round(confs[k], 2),          # YOLO26 confidence that this box is a person
            "face_score": round(float(face.det_score), 2) if face is not None else None,   # SCRFD confidence
            "checked": checked,
            "box": [round(v * back) for v in box],
            "face_box": [round(float(v) * back) for v in face.bbox] if face is not None else None,
        })
    gallery.forget_old(now)

    # 3. the PC's user = the biggest (closest) YOLO person box
    user = None
    if people:
        user = max(people, key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))
        user["user"] = True
    return {"covered": False, "persons": len(people), "people": people, "user": user, "prep": prep}
