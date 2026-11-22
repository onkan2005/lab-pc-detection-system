# PC Usage Monitor

A computer-lab monitor built on **YOLO26**. Each PC has its own camera. The system recognizes who is
sitting at the PC, logs how long they used it, and raises an alert when the person is not assigned to
that PC (only after a time threshold, so sitting down briefly never triggers anything). A web console
shows every PC's camera and an incident feed for the lab technician.

Final project for ICE 415 – Digital Image Processing.

---

## Contents
- [How it works](#how-it-works)
- [Quick start with Docker (recommended)](#quick-start-with-docker-recommended)
- [Run without Docker (Python)](#run-without-docker-python)
- [Enroll students](#enroll-students)
- [Reset the data (keep the photos)](#reset-the-data-keep-the-photos)
- [Same person, not a new one (re-identification)](#same-person-not-a-new-one-re-identification)
- [Using the app](#using-the-app)
- [Cameras (webcam, phone, two cameras)](#cameras)
- [Configuration](#configuration)
- [Course lessons applied](#course-lessons-applied)
- [Privacy](#privacy)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## How it works

```
camera frame (browser) ─► preprocessing ─► YOLO26 + ByteTrack ─► face inside each person box ─► ArcFace
                          (lessons 1–3)     people + track IDs     (SCRFD)                       who is it?
                                                                                                    │
       alert rules with time thresholds ◄── which PC? (camera = PC) ◄───────────────────────────────┘
       (wrong PC, unknown, watchlist, absent, crowding, camera blocked)  ─► incident feed + usage log
```

1. **One camera per PC.** Each camera is mapped to a PC ("this camera is PC-01").
2. **Preprocessing (course lessons).** Brightness and contrast are measured on every frame; dark or flat
   frames are corrected with `g = a·f + b`; a camera blinded by light is detected with the digital negative.
3. **YOLO26 detects and tracks people** (pretrained on COCO, with ByteTrack IDs #1, #2…). The largest
   person box, the one closest to the camera, is the PC's user; others are bystanders.
4. **Face recognition inside each YOLO box.** SCRFD finds the face, ArcFace (InsightFace `buffalo_l`) turns
   it into a 512-number embedding, and cosine similarity matches it with the enrolled students
   (≥ 0.45 = that student, otherwise *unknown*). The name stays attached to the YOLO track ID.
5. **Rules with thresholds** decide when to raise an alert. Every alert becomes an incident with a
   snapshot; the technician marks it *Technician sent* → *Resolved* / *False alarm*.

All models are **pretrained**; no training is needed. Each student only provides photos.

---

## Quick start with Docker (recommended)

Requirements: [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or Docker
Engine (Linux).

```bash
git clone <this-repository-url> pc-usage-monitor
cd pc-usage-monitor
```

1. Add student photos to `enrollment/` ([how](#enroll-students)).
2. Build and start (the first build downloads PyTorch and the models, ~5–10 minutes):
   ```bash
   docker compose up -d --build
   ```
3. Open **http://localhost:8765** in Chrome or Edge.
4. Go to **Students** → **Enroll now** (or **Re-run enrollment**).
5. Go to **PCs** to add PCs and assign students, then **Cameras** to start a camera.

Useful commands:
```bash
docker compose logs -f        # see what the app is doing
docker compose restart        # restart (e.g. after changing config.py and rebuilding)
docker compose down           # stop
docker compose up -d --build  # rebuild after pulling code changes
```

Your data stays on your computer, outside the image:
| On your computer | In the container | What |
|---|---|---|
| `enrollment/` | `/enrollment` (read-only) | student photo folders |
| `detector/data/` | `/data` | face embeddings, PC assignments, watchlist, logs, incident snapshots |

The app is published on `127.0.0.1` only, so other computers on the network cannot open it.

---

## Run without Docker (Python)

Requirements: Python 3.12, about 3 GB of free disk space. On Windows, install Python from python.org and
tick "Add Python to PATH".

```bash
cd detector
python -m venv .venv
```
Activate it: Windows `.venv\Scripts\activate`, Linux/macOS/WSL `source .venv/bin/activate`. Then:
```bash
pip install -r requirements.txt
python web.py
```
Open **http://localhost:8765**. The models download automatically on the first run.

> On Windows, if installing `insightface` fails with a "Microsoft Visual C++" error, install
> [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> ("Desktop development with C++") and run `pip install -r requirements.txt` again.

Command-line alternative (OpenCV window, no browser), one PC per run:
```bash
python monitor.py --pc PC-01 --source 0
```

---

## Enroll students

### 1. Make a folder per student
```
enrollment/
├── juan_dela_cruz/      ← folder name = student's name (shown in the app, lowercased)
│   ├── IMG_2031.jpg     ← any file names
│   ├── photo2.png
│   └── ...
└── maria/
```
- Formats: `.jpg`, `.jpeg`, `.png`. iPhone `.HEIC` is skipped (set the camera to *Most Compatible*).
- Use one folder per person; `Juan` and `juan` would count as the same student.

### 2. Take good photos (10–30 per person)
| Shot | How many | Notes |
|---|---|---|
| Front, looking at the camera | 4–5 | neutral face, good light in front (no window behind) |
| Turned slightly left / right (~15–30°) | 2 each | both eyes still visible; no full side view |
| Looking slightly up / down | 2 each | "down" = like reading the keyboard |
| From the real PC camera position | 3–5 | the angle the system will actually see |
| With glasses / cap, if worn | 1–2 | |

Avoid: blurry shots, other people in the frame, a hand over the face, filters/beauty mode.
More photos from **different angles and lighting** help; many copies of the same pose do not.

### 3. Enroll
New folders and added/removed photos are **detected automatically**: the **Students** page shows
"1 change found …" and a badge appears on the Students icon. Click **Enroll now**.
(Command line: `python enroll.py` from the `detector` folder.)

The Students page then shows each student's best photo, how many photos were usable, and with
**Show all photos** which ones were weak or skipped (green = used, orange = weak, red = no face found).

Use **Check a photo** (Students page) to test recognition: it shows the similarity to every enrolled
student and the threshold line.

---

## Reset the data (keep the photos)

Wipes PCs and assignments, watchlist, incidents and snapshots, usage/alert logs and the old face
embeddings, **keeps every photo in `enrollment/`**, and re-enrolls everyone. It asks for confirmation
and refuses to run while the app is running (the app would write the old data back).

With Docker:
```bash
docker compose down
docker compose run --rm pc-usage-monitor python reset.py
docker compose up -d
```

Without Docker (from the `detector` folder, app stopped):
```bash
python reset.py
```

Options: `--keep-pcs` keeps PCs, assignments and the watchlist (clears only incidents, logs and
enrollment) · `--yes` skips the confirmation · `--no-enroll` doesn't re-enroll afterwards.

To clear **only the incident feed**, delete `detector/data/incidents.json` and `detector/data/snapshots/`
while the app is stopped.

---

## Using the app

| Page | What you do there |
|---|---|
| **Overview** | Camera wall (one tile per PC, filter by *Needs attention / Online / Offline*) and the incident **Feed**. Click **Send technician**, then **Resolved** or **False alarm**. |
| **Cameras** | Choose *This camera is PC-0X* and the camera device, **Start camera**. **Add camera** for a second camera on the same computer. Under the video: who is there, the verdict, image brightness/contrast. |
| **Students** | Enrolled students, **Watchlist** checkbox (banned from all PCs), enrollment, **Check a photo**. |
| **PCs** | Every PC, camera status, who is using it now, assigned students (**Edit** opens a searchable list; a PC can have several students), add/remove PCs, usage log. |
| **How it works** | Pipeline, alert thresholds, course lessons table, **Image processing lab** (upload a photo, see every lesson's operation with its histogram, simulate a dim room). |

Rail buttons: alarm sound on/off, light/dark mode.

What the colors mean on the video: **green** = assigned to this PC, **orange** = not assigned, timer running,
**red** = alert raised, **gray** = bystander or face not visible yet.

---

## Cameras

The camera is opened **by the browser** (so it also works when the server runs in Docker or WSL), and
frames are sent to the server for detection.

- **Laptop webcam:** pick it in *Camera device*.
- **Phone as a camera:** install [Iriun Webcam](https://iriun.com/) (or DroidCam) on the phone and the PC,
  same Wi-Fi; then pick "Iriun Webcam" in *Camera device*.
- **Two cameras = two PCs on one computer:** e.g. laptop webcam as PC-01 and Iriun as PC-02.
  On **Cameras** click **Add camera** and give each panel its own PC and device.
- Camera access only works on `http://localhost` (or HTTPS). Click **Allow** when the browser asks.

---

## Configuration

All settings are in [`detector/config.py`](detector/config.py) (restart, or rebuild the Docker image,
after changing them):

| Setting | Default | Meaning |
|---|---|---|
| `UNAUTHORIZED_SEC` | 60 | unknown / unassigned person at a PC → alert after |
| `WRONG_SEAT_SEC` | 60 | student at someone else's PC → alert after |
| `WATCHLIST_SEC` | 3 | watchlisted person at any PC → alert after |
| `ABSENT_SEC` | 600 | none of a PC's assigned students there → alert after (camera online only) |
| `CROWD_COUNT`, `CROWD_SEC` | 3, 30 | this many people at one PC for this long → alert |
| `TAMPER_SEC` | 5 | camera covered / blank / blinded → alert after |
| `MATCH_THRESHOLD` | 0.45 | face similarity needed to count as a student |
| `ENHANCE_METHOD` | `brightness` | low-light correction: `brightness`, `normalize`, `equalize`, `clahe` |
| `REID_WINDOW_SEC` | 300 | how long a camera remembers people's clothes/body for re-identification |
| `REID_MIN_SIM`, `REID_SPLIT_SIM` | 0.70, 0.25 | appearance similarity to call it the same person / a different person on the same track |
| `PRIVACY_BLUR` | `True` | pixelate faces in the camera wall, incident snapshots and live view |
| `RETENTION_DAYS` | 30 | delete finished incidents, snapshots and log rows older than this (0 = keep) |

For a demo, lower the alert times (e.g. 10 seconds) so you don't have to wait.

Environment variables (used by Docker): `PCMON_DATA_DIR`, `PCMON_ENROLLMENT_DIR`, `PCMON_HOST`, `PCMON_PORT`.

---

## Course lessons applied

| Lesson | Where it is used |
|---|---|
| Definitions and Notations | images as f(x, y), M×N pixels, intensities r ∈ [0, L−1], L = 256 (`preprocess.py`) |
| Image Data Formats | JPEG → 8-bit BGR array → grayscale / YCrCb (`detection.py`, `preprocess.py`) |
| L1 Brightness and Contrast | brightness = mean, contrast = std on every frame; dark/flat frames corrected with `g = a·f + b`; contrast drives the camera-blocked check |
| L2 Digital Negative and Normalization | negative `s = (L−1) − r` detects a camera blinded by light; min-max normalization is an alternative |
| L3 Histogram Equalization | `s_k = (L−1)·CDF(r_k)` written from the formula (verified equal to OpenCV) and compared by experiment |
| Generating Outputs | annotated video, incident snapshots, CSV logs, experiment table, histograms |
| Beyond the course | YOLO26 + ByteTrack, ArcFace embeddings + cosine similarity, CLAHE, threshold rules |

**Which low-light correction is used was decided by experiment.** From the `detector` folder:
```bash
python evaluate_lighting.py
```
It darkens each enrolled student's photo from 100% to 2.5% light (with sensor noise), applies every
method, runs YOLO26 + recognition, and writes `data/lighting_results.csv`. In our runs `g = a·f + b` gave
the highest similarity at every dim level, normalization was close, and histogram equalization / CLAHE
scored lowest because they also amplify sensor noise. Re-run it after everyone is enrolled.

---

## Same person, not a new one (re-identification)

YOLO's tracker gives a person a **new track ID** whenever they leave the frame, are blocked or turn
away. Without help, the same student would keep "appending" as Person #1, #2, #3… and lose their name
while their face is hidden. `reid.py` keeps a **short-term appearance memory** per camera:

- **Clothing colors**: hue/saturation histograms (HSV) of the shirt, and the pants when the whole body is
  visible. HSV separates color from brightness, so lighting changes matter less. The face area is left out.
- **Body dimensions**: proportions (height/width) and size of the YOLO person box.
- A new track that looks like someone seen in the last 5 minutes **is** that person: same number, same
  name. The video label then says e.g. `labisores (by blue clothes)`.
- If the tracker hands a seat to the **next** person (it links boxes by position), the clothes no longer
  match and that is counted as a different person.
- **The face always wins.** Appearance is only short-term (clothes change every day). A clear face that
  says it is someone else splits them apart (e.g. same school uniform). A **blurred or turned face**
  (measured with image sharpness, the variance of the Laplacian) does not override who the clothes say it is.

Tested with two enrolled students: the tracker produced 4 different track IDs, re-identification
merged them into the right 2 people and kept the names while faces were blurred. Thresholds are in
`config.py` (`REID_*`, `FACE_SHARPNESS_MIN`); re-check them after the whole group is enrolled.

---

## Privacy

- **Faces are blurred for admins.** With `PRIVACY_BLUR = True` (default), faces are pixelated in the camera
  wall, in incident snapshots (blurred **before** saving, so the sharp image is never stored) and in the live
  camera view. If no face is visible, the head area is blurred. The name label still tells the technician
  who it is. Tested: the face detector cannot find a face in the stored snapshots or thumbnails.
- **Automatic deletion.** Finished incidents, their snapshots and log rows older than `RETENTION_DAYS`
  (default 30) are deleted automatically; open incidents are kept until closed.
- **Not committed, not in the image.** Photos, embeddings, names, logs and snapshots stay in `enrollment/`
  and `detector/data/`, which are excluded from Git and Docker (`.gitignore`, `.dockerignore`). The app is
  only reachable from the computer it runs on.
- **What about hashing?** Hashing fits some data, not all of it:
  - Face **embeddings cannot be hashed**: matching measures how close two faces are, and a hash destroys
    that closeness. The right protection is **encryption at rest** and access control.
  - **Names in logs can be pseudonymized** with a keyed hash (HMAC), so a leaked log shows codes instead of names.
  - **Photos** are only needed to build the embeddings; they can be deleted after enrollment.
  These are possible next steps, not implemented yet. The app also has **no admin login** yet; anyone at the
  computer running it can open the console.
- Get each person's consent before enrolling them (Philippine Data Privacy Act, RA 10173).
- The InsightFace pretrained models are licensed for non-commercial research use, which fits a
  school project.

---

## Project structure

```
├── Dockerfile, docker-compose.yml   container setup
├── enrollment/                      student photo folders (not committed)
└── detector/
    ├── web.py                       web server (Flask) + API
    ├── templates/index.html         the web app (single page)
    ├── detection.py                 per-frame pipeline: preprocessing → YOLO26 track → re-ID → face → ArcFace
    ├── reid.py                      re-identification by clothing colors + body dimensions
    ├── preprocess.py                course lessons: brightness/contrast, negative, normalization, equalization
    ├── faces.py                     face detection + recognition (cosine similarity)
    ├── tracker.py                   usage sessions and alert rules
    ├── drawing.py                   labels and drawing
    ├── incidents.py                 incident feed with snapshots
    ├── enroll.py                    builds the face embeddings
    ├── reset.py                     wipes app data, keeps photos, re-enrolls
    ├── evaluate_lighting.py         low-light experiment
    ├── monitor.py, find_camera.py   command-line version and camera finder
    ├── config.py                    all settings
    └── data/                        runtime data (not committed)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Access to localhost was denied" / HTTP 403 on port 5000 | Windows uses port 5000; this app uses **8765** |
| Camera list is empty or camera won't start | open the app at `http://localhost:8765` (not an IP address), click **Allow**; make sure no other app is using the camera |
| A student is always "unknown" | add more photos from the real camera angle, re-enroll; check the photo with **Check a photo** |
| Someone is recognized as the wrong student | add more varied photos of both; raise `MATCH_THRESHOLD` slightly (e.g. 0.5) |
| "Camera blocked" in a dark room | the camera only counts as blocked if there is almost no detail **and** YOLO sees nobody; add light |
| Slow detection | close other heavy apps; detection runs on the CPU |
| Port 8765 already in use | stop the other copy (`docker compose down`), or change `PCMON_PORT` and the port in `docker-compose.yml` |
