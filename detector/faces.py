"""Face detection + recognition using pretrained InsightFace (SCRFD + ArcFace)."""
import os

import cv2
import numpy as np
import onnxruntime as ort
import torch
from insightface.app import FaceAnalysis

import config


def load_face_app(det_size=config.FACE_DET_SIZE):
    app = FaceAnalysis(name=config.FACE_MODEL, allowed_modules=["detection", "recognition", "landmark_2d_106"],   # landmarks: blink check
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    # ONNX Runtime keeps its threads busy-waiting after each run, which starves YOLO (PyTorch)
    # running right after it. Re-open the sessions with spinning off and a share of the CPU cores.
    opts = ort.SessionOptions()
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
    for model in app.models.values():
        model.session = ort.InferenceSession(model.model_file, sess_options=opts, providers=["CPUExecutionProvider"])
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    return app


def detect_faces(app, img):
    """Face detection that also works on close-ups.

    At this detector size a face filling most of a selfie can be missed.
    If nothing is found, add a border so the face takes up less of the image, and try again.
    """
    faces = app.get(img)
    if faces:
        return faces
    h, w = img.shape[:2]
    pad = max(h, w) // 2
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    faces = app.get(padded)
    for f in faces:
        f.bbox = f.bbox - np.array([pad, pad, pad, pad], dtype=f.bbox.dtype)
    return faces


class Recognizer:
    """Compares a face embedding with every enrolled embedding (cosine similarity)."""

    def __init__(self, path=config.EMBEDDINGS_FILE):
        data = np.load(path)
        self.embeddings = data["embeddings"]          # (N, 512), already L2-normalized
        self.names = list(data["names"])               # name for each row

    def identify(self, embedding):
        sims = self.embeddings @ embedding             # cosine similarity per enrolled photo
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score < config.MATCH_THRESHOLD:
            return "unknown", score
        return self.names[best], score
