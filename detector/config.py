"""All tunable settings for the PC usage detection system.

Setup: one camera per PC. Each camera is mapped to a PC (e.g. "this camera is PC-01"),
so whoever sits in front of that camera is the user of that PC.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# Personal data (photos, face embeddings, logs, snapshots, who is assigned where) lives outside the
# code so it is never committed or baked into the Docker image. Both folders can be moved with
# environment variables (the Docker setup mounts them as volumes).
DATA_DIR = Path(os.environ.get("PCMON_DATA_DIR", BASE_DIR / "data"))
ENROLLMENT_DIR = Path(os.environ.get("PCMON_ENROLLMENT_DIR", BASE_DIR.parent / "enrollment"))  # <name>/*.jpg
DATA_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npz"
ENROLL_REPORT_FILE = DATA_DIR / "enroll_report.json"   # per-photo quality + each person's best photo
ASSIGNMENTS_FILE = DATA_DIR / "assignments.json"  # {"PC-01": ["juan", "maria"], ...}; also the list of PCs
WATCHLIST_FILE = DATA_DIR / "watchlist.json"      # enrolled people who must not use any PC
INCIDENTS_FILE = DATA_DIR / "incidents.json"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
SESSIONS_LOG = DATA_DIR / "sessions.csv"
ALERTS_LOG = DATA_DIR / "alerts.csv"

# --- Models (both pretrained, downloaded automatically on first run) ---
YOLO_MODEL = "yolo26n.pt"          # COCO-pretrained YOLO26: counts the people in front of the PC
FACE_MODEL = "buffalo_l"           # InsightFace: SCRFD face detector + ArcFace recognizer
YOLO_CONF = 0.4
FACE_DET_SIZE = 640                # enrollment / photo test (any photo)
# Live: the PC camera is close to the user, so faces are big and small detector inputs are enough.
YOLO_IMGSZ = 320
LIVE_FACE_DET_SIZE = 320
DETECT_WIDTH = 640                 # frames are shrunk to this width before detection
# Recognition (ArcFace) is the slowest step. A face that stays in the same spot keeps its name
# and is only re-recognized every RECOGNIZE_EVERY seconds.
RECOGNIZE_EVERY = 1.0
FACE_TRACK_IOU = 0.4               # same face as last frame if the boxes overlap this much

# --- Preprocessing (course lessons: brightness/contrast, negative/normalization, histogram equalization) ---
PREPROCESS = True         # enhance dark / flat frames before YOLO26 and face recognition
DARK_MEAN = 60            # mean intensity below this = too dark -> enhance
LOW_CONTRAST_STD = 25     # intensity std dev below this = low contrast -> enhance
GLARE_MEAN = 235          # mean intensity above this = over-exposed / camera blinded by a light
# Which enhancement to use on dark/flat frames. Chosen by experiment (python evaluate_lighting.py,
# results in data/lighting_results.csv): the Lesson 1 linear brightness/contrast stretch gave the
# highest ArcFace similarity at every dim level (45% .. 8% light), min-max normalization was a close
# second, and histogram equalization / CLAHE scored lowest because they also amplify sensor noise.
ENHANCE_METHOD = "brightness"  # "brightness" (Lesson 1) | "normalize" (Lesson 2) | "equalize" (Lesson 3) | "clahe"

# --- Face recognition ---
MATCH_THRESHOLD = 0.45   # cosine similarity; below this the face is "unknown"
MIN_FACE_PX = 50         # faces narrower than this (in the 640-wide frame) are ignored: too far away

# --- Thresholds (seconds) ---
PRESENCE_GAP = 3          # not seen for this long = left the PC (for alert timers)
SESSION_GAP = 120         # not seen for this long = usage session ends and is logged
CAMERA_OFFLINE_SEC = 10   # no frames from a PC's camera for this long = camera offline
TAMPER_STD = 4            # detail (std dev of the blurred grayscale) below this AND no person seen = covered
TAMPER_SEC = 5            # camera covered this long -> CAMERA_BLOCKED incident
WATCHLIST_SEC = 3         # watchlisted person at any PC (short: just enough to rule out a glitch)
UNAUTHORIZED_SEC = 60     # unknown / unassigned person using a PC
WRONG_SEAT_SEC = 60       # assigned student using someone else's PC
ABSENT_SEC = 600          # none of a PC's assigned students at it (only while that PC's camera is online)
CROWD_COUNT = 3           # this many people in front of one PC ...
CROWD_SEC = 30            # ... for this long = crowding alert
