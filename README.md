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
- [Far-away faces: many frames instead of one](#far-away-faces-many-frames-instead-of-one)
- [Photo or real face? (liveness)](#photo-or-real-face-liveness)
- [How accurate is it?](#how-accurate-is-it)
- [Using the app](#using-the-app)
- [Cameras (webcam, phone, two cameras)](#cameras)
- [CCTV room camera](#cctv-room-camera)
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
5. **Liveness.** A face shown on a phone or screen (YOLO26 also detects phones and screens) is never given
   a name, and a recognized face must blink (106 face landmarks) or it is flagged as a possible photo.
6. **Rules with thresholds** decide when to raise an alert. Every alert becomes an incident with a
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
embeddings, **keeps every photo in `enrollment/`**, and re-enrolls everyone. It asks for confirmation.
The app may keep running: within a second it notices the reset, clears the incident feed, timers and
visits it still had in memory, and loads the fresh data (the screen empties by itself).

With Docker (app running):
```bash
docker compose exec pc-usage-monitor python reset.py
```

Without Docker (from the `detector` folder):
```bash
python reset.py
```

Options: `--keep-pcs` keeps PCs, assignments and the watchlist (clears only incidents, logs and
enrollment) · `--yes` skips the confirmation · `--no-enroll` doesn't re-enroll afterwards.

To clear **only the incident feed**, delete `detector/data/incidents.json` and `detector/data/snapshots/`
while the app is stopped (or restart it afterwards: `docker compose restart`).

---

## Using the app

| Page | What you do there |
|---|---|
| **Overview** | Camera wall (one tile per PC, filter by *Needs attention / Online / Offline*) and the incident **Feed**. Click **Send technician**, then **Resolved** or **False alarm**. |
| **Cameras** | Choose *This camera is PC-0X* and the camera device, **Start camera**. **Add camera** for a second camera on the same computer. Under the video: who is there, the verdict, image brightness/contrast. |
| **CCTV** | Room cameras (**Add camera** for more): everyone in view, who is in the room and for how long, room log, the head-pose/distance test results. |
| **Students** | Enrolled students, **Watchlist** checkbox (banned from all PCs), enrollment, **Check a photo**, **Check a video** (see below). |
| **PCs** | Every PC, camera status, who is using it now, assigned students (**Edit** opens a searchable list; a PC can have several students), add/remove PCs. |
| **Usage log** | Every visit of every person to every PC camera (see below): tabs, filters, totals, **Export CSV**. |
| **How it works** | Pipeline, alert thresholds, course lessons table, **Image processing lab** (upload a photo, see every lesson's operation with its histogram, simulate a dim room). |

Rail buttons: alarm sound on/off, light/dark mode.

No refreshing needed: every page updates itself. Changes to PCs, students, the watchlist or incidents show on
all open pages within a second; after an app update or restart the page reloads by itself and cameras that
were running start again. Deleting a PC removes it from the dashboard at once; a camera panel that was on it
stops and moves to a free PC instead of bringing the deleted PC back. Cameras are numbered by their place on the
page: remove cameras 1–4 of 5 and the one left becomes Camera 1 on PC-01 (only PCs without students are
renamed). A camera added or removed in one tab also shows in another open tab of the Cameras page.

**Usage log.** One row per person per visit, for everyone a PC camera sees, sorted into tabs:
**PC use** (a student in the PC's seat: the counted usage), **Unknown at a PC** (not enrolled, or face never
seen), **Bystanders** (next to or behind the user) and **Photo / phone** (a face shown on a device). Each row
has start, end, duration and active time (h:mm:ss), breaks and time away, own PC or not, the PCs the student
is assigned to, how they were identified (face / clothes / tracking, in % of the frames), the best face match
and the alerts raised during the visit. Filters: name, PC, date range, own PC, minimum length, in progress or
finished, identified by, with/without alerts, and sort order. **Export CSV** saves exactly what is shown.

A visit ends when the person has been gone for `SESSION_GAP` (2 minutes). Leaving for 30 minutes and coming
back is therefore a **new row**; a shorter absence stays in the same row and is counted as a break (with the
time away, so *active* = duration − breaks). Visits shorter than 3 s (someone walking past) are not logged.
An old `sessions.csv` is converted to the new columns automatically.

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

## CCTV room camera

Besides one camera per PC, the **CCTV** page takes a camera that watches the **whole room** (for example
the phone with Iriun on a high shelf). Everyone in view is detected by YOLO26 and recognized; there is no
"PC user" here, it answers *who is in the room*.

**Add camera** adds more room cameras (ROOM-1, ROOM-2, …, each with its own name, webcam, minimize/maximize
and remove), for example one in each corner. They share the room list and the room log; a student seen by any
of them is "in the room".

- Frames are sent at 1280 px wide and YOLO26 runs at 960 px, so small, far-away people and faces are found
  (slower than a PC camera: about 0.2 s per frame on a laptop CPU).
- **In the room:** each recognized student, when they entered and how long they stayed. A visit ends after
  `CCTV_LOG_GAP` seconds unseen and is written to `data/room_log.csv`.
- **Incidents:** a watchlisted student in the room, an unknown person in the room for `CCTV_UNKNOWN_SEC`
  seconds (UNKNOWN IN ROOM), the camera blocked.
- **Not looking at the camera?** A face turned left/right is still recognized; looking down is the
  weakest pose. The back of the head cannot be recognized by anyone, so a person is then named only
  if their face was seen earlier and their clothes and body still match (shown as *"by red clothes"*);
  otherwise they stay *"face not visible"*.

How well does it work? Run the test (results are also shown on the CCTV page):

```bash
python evaluate_cctv.py
```

It shrinks every enrolled photo so the face is 160 px (close) down to 24 px (far) wide, places it in a
1280×720 frame and recognizes it against the *other* photos. Measured on 29 photos of 3 students:

| Face width | 160 px | 80 px | 48 px | 32 px | 24 px |
|---|---|---|---|---|---|
| Recognized correctly | 97% | 97% | 97% | 90% | 69% |
| Wrong person | 0 | 0 | 0 | 0 | 0 |

**Check a photo** (Students page) runs one photo through the same steps and shows the numbers:
YOLO26 people (confidence), SCRFD faces with their 5 landmarks, the ArcFace embedding, the cosine
similarity to every enrolled person against the 0.45 threshold and the margin to the next person, the
time each step took, the photo's brightness/contrast and histogram, and for each face what decides
whether a live camera can recognize it: face size (PC ≥ 50 px, CCTV ≥ 24 px), sharpness, head pose
(yaw/pitch from the landmarks) and whether it is inside a YOLO26 person box.

**Test on a real video** (does a person keep their name when the face is gone?). In the app:
**Students → Check a video**. Upload a recording, choose the camera type, optionally tick who is really
in it, and press **Start test**. While it runs you see, live:

- the current frame with YOLO26 boxes and labels (faces pixelated),
- **time per frame for each step**: preprocessing, YOLO26 + tracker, re-ID, SCRFD face detection, ArcFace,
- **the numbers behind every decision** in the current frame: brightness/contrast and the correction, the
  YOLO26 input size and each box's confidence, the face size and detector score, the ArcFace similarity
  compared with the 0.45 threshold, and the re-ID similarity when someone is kept by their clothes,
- running totals: people at once, faces found, identities, number changes, alerts, and (if you ticked
  who is in the video) right / unknown / **wrong** names with and without a visible face.

At the end the face-blurred result video plays on the page. The uploaded original is deleted after
the test; results are kept in `data/video_tests/` (last 5 tests). One test runs at a time, and live
cameras get slower while it runs (it uses the same CPU). The same test from the command line:

```bash
python evaluate_video.py ../test_videos/my_clip.mp4 --expect juan,maria       # your own recording
python evaluate_video.py ../test_videos/faceocc2.webm --enroll tester --fps 0  # public sample, person enrolled temporarily, every frame
```

It saves `result.webm` and `summary.json` in `<video>_result/` (keep test videos in `test_videos/`,
ignored by Git and Docker). Results on public test videos:

| Video | What it tests | Result |
|---|---|---|
| OpenCV `faceocc2`, face covered for 20 s, person leaves and returns | name kept without a face | **100%** of 473 no-face frames had the right name, 0 wrong names |
| EPFL lab, 6 people walking for 2 min (360×288 CCTV) | tracking in a crowded room | faces too small to recognize (found in 1% of boxes); 45 identities for 6 people (42 before the look-alike rule, which refuses merges that are too close to call); 7 "unknown in room" alerts for 6 strangers |
| CAVIAR shop corridor, person leaves and re-enters (384×288) | re-entry | 2 identities for 2 people |

What this means: when people walk behind each other for seconds, nobody can be followed by motion alone;
the system then needs a face (or very distinct clothes) to know who it is again. It never guesses: in
all tests no one was ever given another person's name. For a lab, use a camera of 1080p placed so that
faces are at least ~32 px wide (see the table above), so names come back whenever someone looks up.

Turned faces (left/right) were recognized at every size; looking down failed most when far away. Faces
narrower than about 32 px (a person across a big room on a 720p camera) are unreliable: use a
1080p camera or put it closer.

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
| `SESSION_GAP` | 120 | gone this long = the visit ends; coming back starts a new usage log row |
| `MATCH_THRESHOLD` | 0.45 | face similarity needed to count as a student |
| `ENHANCE_METHOD` | `brightness` | low-light correction: `brightness`, `normalize`, `equalize`, `clahe` |
| `REID_WINDOW_SEC` | 300 | how long a camera remembers people's clothes/body for re-identification |
| `REID_MIN_SIM`, `REID_SPLIT_SIM` | 0.70, 0.25 | appearance similarity to call it the same person / a different person on the same track |
| `REID_MARGIN` | 0.05 | two recent people match this closely: too close to call, wait for the face |
| `LIVENESS`, `LIVENESS_SEC`, `SPOOF_SEC` | `True`, 20, 2 | blink check and phone/screen check; no blink for this long / face on a phone this long → alert |
| `FUSION`, `FUSE_FRAMES`, `FUSE_MIN_FACE_PX` | `True`, 20, 16 | multi-frame face average: on/off, frames used, smallest face collected (CCTV) |
| `HINT_SEC`, `HINT_SIM` | 180, 0.60 | cross-camera "maybe <name>?" hints: how recent, how similar the clothes |
| `ROOM_NOTE_SEC` | 60 | an assigned student seen by another camera this recently → note on their empty PC |
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
- **Clothes that change while the person stays put** (jacket on/off, a light switched on) do not make
  them a new person: if the box is still in the same spot as in the previous frame, it is the same person
  and the new colors are learned quickly. Swapping people takes seconds and moves the boxes, so that is
  still caught. Test (enrolled student, face hidden, shirt suddenly turns dark red while seated): name
  kept in **100%** of the 51 no-face frames, against 27.5% before this rule.
- **Look-alikes:** if two recent people both match and are within 0.05 of each other (similar outfits),
  the system does not pick one; the person stays unnamed until their face is seen.
- **Between cameras:** each camera keeps its own memory (colors look different from another angle), but a
  student recognized by face on one camera is shared with the others (`sightings.py`). A person whose face
  was not seen yet, whose clothes clearly match that student, is shown as `maybe <name>?`. It is only a
  hint: it is never logged and never raises or clears an alert. The same board gives the PC tiles a note
  such as *"juan is in the room (CCTV cam1, 12s ago), not at this PC"*.
- **The face always wins.** Appearance is only short-term (clothes change every day). A clear face that
  says it is someone else splits them apart (e.g. same school uniform). A **blurred or turned face**
  (measured with image sharpness, the variance of the Laplacian) does not override who the clothes say it is.

**How people are numbered on screen.** Enrolled students are shown by name. Anyone else is
`Unknown person N`, where N stays with that person while they are in view and is reused by the next
person after they leave (`labels.py`). So the numbers stay as small as the number of people around and
do not jump when people walk past each other. On the EPFL 6-person lab video: highest number shown 7
(at most 6 people in view), and a person's number changed **0 times** while they stayed in view
(numbering by position in the frame would have changed 864 times).

Tested with two enrolled students: the tracker produced 4 different track IDs, re-identification
merged them into the right 2 people and kept the names while faces were blurred. Thresholds are in
`config.py` (`REID_*`, `FACE_SHARPNESS_MIN`); re-check them after the whole group is enrolled.

---

## Far-away faces: many frames instead of one

On a CCTV camera a student's face is often only 16–30 pixels wide. One such frame is too noisy for ArcFace,
but YOLO26's tracker knows that the faces in consecutive frames belong to the **same person**. `fusion.py`
keeps the face embeddings of each tracked person (last 20 frames, 10 s), averages them (bigger and more
confident faces count more) and matches the **average** with the same 0.45 threshold. The noise of single
tiny frames cancels out. Faces from 16 px are collected for the average; one face alone still needs 24 px.

Safety: the samples are only used while the person has no name yet, they are thrown away when re-ID joins a
track to a person by clothes, and when a new sample looks like a different face (the tracker swapped people).

Measured with `python evaluate_fusion.py` (every enrolled photo shrunk to 16–32 px and seen over 15 frames
with movement, blur, noise and JPEG; matched against the *other* photos; strangers = that student removed):

| Face size | Students named, one frame at a time (before) | With the multi-frame average | Strangers given a name (before / after) |
|---|---|---|---|
| 16 px | 0% | **17%** | 0 / 0 of 29 |
| 20 px | 0% | **58%** | 0 / 0 of 29 |
| 24 px | 68% (after 3.1 frames) | **72%** (after 2.0 frames) | 0 / 0 of 29 |
| 28 px | 82% | 82% | 0 / 0 of 29 |
| 32 px | 86% | 86% | 0 / 0 of 29 |

No wrong names at any size, before or after. A student walking toward the camera (face growing from 14 to
34 px over 12 s on a lab background) was named 1.6 s earlier (83% of frames named instead of 65%) when YOLO26
found the person early. When YOLO26 found the person only once they were close, both versions were the same:
without a YOLO26 person box there is no face to collect (see *How it works*).

On the real test videos nothing got worse: faceocc2 (CCTV and PC) and CAVIAR gave exactly the same results.
On the EPFL lab video (6 strangers, tiny faces) faces were now collected in 12.8% of person boxes instead of
0.8%, and still **nobody was given a student's name**; identities and alerts were unchanged.

---

## Photo or real face? (liveness)

ArcFace only compares faces, so a printed photo or a phone showing a student's face would match that
student. `liveness.py` adds two checks, both with pretrained models:

1. **Face on a phone or screen (YOLO26).** YOLO26 also looks for COCO *cell phone*, *tv* and *laptop*. A face
   inside a phone, or a person entirely inside a screen, is being shown to the camera: it gets no name and,
   if it looks like an enrolled student for 2 s, raises a **SPOOF** incident.
2. **Blinking (InsightFace 106 landmarks, PC cameras).** Eye openness = gap between the eyelids ÷ distance
   from the eyes to the mouth (so tilting a photo does not fake a blink). A blink is openness below 60% of
   that person's usual value. A recognized PC user whose face is visible for 20 s without one blink raises
   **NO_BLINK** ("a printed photo?").

Measured:

| Test | Result |
|---|---|
| Enrolled photo held up and moved, tilted ±25°, for 30 s | eye openness changed only ±10%, **0 blinks**; NO_BLINK alert at 20 s |
| Real face (`faceocc2`, 32 s, partly covered by a book) | first blink after 1 s, **no false alert** |
| Cost | about 6–9 ms per frame (blink check) |

Limits: a **video** of a student played on a phone does blink, so it is only caught when YOLO26 sees the
phone. Blinks are not measured on CCTV (faces too small for eyelid landmarks). Test the phone check with a
real phone in front of the webcam: a drawn phone in a test video was not recognized as a phone by YOLO26.

---

## How accurate is it?

**Students page → Recognition accuracy → Measure accuracy** (or `python evaluate_accuracy.py`) measures
face recognition on the enrolled photos, with nothing new to collect:

- each photo is recognized against all the **other** photos and should get its own name (duplicates of
  the same shot are left out);
- each student in turn is **removed** from the system, so their photos are strangers and should be rejected.

Put face photos of people who are **not** enrolled in `data/strangers/` for a bigger stranger test. Results
with 3 students, 29 photos:

| Threshold | Recall (students named right) | False accepts (strangers named) | Wrong student's name | Precision |
|---|---|---|---|---|
| 0.25 | 100% | 0% | 0% | 100% |
| 0.35 | 96.6% | 0% | 0% | 100% |
| **0.45 (used)** | **96.6%** | **0%** | **0%** | **100%** |
| 0.55 | 93.1% | 0% | 0% | 100% |
| 0.70 | 69.0% | 0% | 0% | 100% |

The highest stranger score was 0.235 and the lowest same-student score 0.334, so 0.45 leaves a margin of
0.21 above the most similar stranger. The one miss at 0.45 is a single unusual photo that still scored 0.33.
29 stranger tests is a small sample, so the threshold is kept well above the highest stranger score
instead of being lowered to the edge. The page also shows the confusion matrix and the score distributions
(`data/accuracy/accuracy.png` from the command line).

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
    ├── liveness.py                  photo/phone check: YOLO26 phones/screens + blink detection
    ├── sightings.py                 shared between cameras: hints and "student is in the room"
    ├── evaluate_accuracy.py         recall, false accepts, precision, confusion matrix per threshold
    ├── fusion.py                    multi-frame face fusion (average of a tracked person's faces)
    ├── evaluate_fusion.py           small faces: one frame vs the multi-frame average
    ├── preprocess.py                course lessons: brightness/contrast, negative, normalization, equalization
    ├── faces.py                     face detection + recognition (cosine similarity)
    ├── tracker.py                   alert rules (PC cameras)
    ├── usage.py                     usage log: every visit of every person, with breaks
    ├── room.py                      room presence, room log and alerts (CCTV cameras)
    ├── drawing.py                   labels and drawing
    ├── incidents.py                 incident feed with snapshots
    ├── enroll.py                    builds the face embeddings
    ├── reset.py                     wipes app data, keeps photos, re-enrolls
    ├── evaluate_lighting.py         low-light experiment
    ├── evaluate_cctv.py             head pose × distance recognition test
    ├── photocheck.py                Check a photo: every step timed, every measurement
    ├── videotest.py                 video test with live stats (Check a video page)
    ├── evaluate_video.py            the same video test from the command line
    ├── labels.py                    stable 'Unknown person N' numbers
    ├── tracker_cctv.yaml            BoT-SORT tracker settings for CCTV cameras
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
