"""Multi-frame face fusion: recognize a person from many small glimpses instead of one good frame.

A CCTV camera often sees a student's face at only 16-30 pixels. One such frame is too noisy for ArcFace,
but the YOLO26 tracker knows that the faces in consecutive frames belong to the SAME person. So each new
face embedding of a tracked person is kept (last FUSE_FRAMES samples within FUSE_SEC), and their weighted
average (bigger and more confident faces count more) is matched against the enrolled students, with the
same MATCH_THRESHOLD as a single face. Averaging cancels much of the per-frame noise (blur, JPEG, pixels).

Safety (a wrong name is worse than no name):
  - the samples are only used while the person has no name yet; a name still needs MATCH_THRESHOLD;
  - the samples are thrown away when re-ID joins a track to a person by clothes (reid.py), and when a new
    sample looks like a different face than the average (FUSE_CONSISTENT), e.g. the tracker swapped people.
"""
import numpy as np

import config


def weight(det_score, face_px):
    """How much one face sample counts: detector confidence x size (faces of 32 px and more count fully)."""
    return float(det_score) * min(1.0, face_px / 32.0)


def add(person, emb, w, now):
    """Adds one face embedding of this person. Returns (average embedding, number of samples)."""
    buf = person.setdefault("face_buf", [])
    buf[:] = [s for s in buf if now - s[2] <= config.FUSE_SEC]
    if len(buf) >= config.FUSE_MIN:
        mean = average(buf)
        if float(mean @ emb) < config.FUSE_CONSISTENT:   # not the same face: the tracker swapped people
            buf.clear()
    buf.append((np.asarray(emb, dtype=np.float32), w, now))
    del buf[:-config.FUSE_FRAMES]
    return average(buf), len(buf)


def average(buf):
    v = np.sum([w * e for e, w, _ in buf], axis=0)
    return v / max(1e-9, float(np.linalg.norm(v)))


def forget(person):
    person.pop("face_buf", None)
