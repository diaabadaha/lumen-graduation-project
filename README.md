# Lumen

**A task-oriented assistive-vision system that helps blind users find objects in their
environment, navigate to landmarks, and reach for objects — using only a smartphone.**

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/server-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/vision-YOLOv8-orange.svg)](https://docs.ultralytics.com/)
[![MediaPipe](https://img.shields.io/badge/hands-MediaPipe-4285F4.svg)](https://developers.google.com/mediapipe)
[![Whisper](https://img.shields.io/badge/STT-faster--whisper-7c3aed.svg)](https://github.com/SYSTRAN/faster-whisper)
[![Door mAP@50](https://img.shields.io/badge/door%20detector%20mAP%4050-0.946-brightgreen.svg)](#the-trained-door-detector)
[![Tests](https://img.shields.io/badge/tests-280%2B%20passing-brightgreen.svg)](#running-tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)

> Birzeit University — ENCS5200 Graduation Project — 2026.

---

## What is Lumen?

A blind user opens a web page on their phone, points the camera at the room, and says
**"find my cup."** Lumen answers in voice: *"Looking for your cup."* As they pan the
camera, Lumen tracks the cup and calls out direction and distance ("to your left, a few
steps away" → "straight ahead, close by" → "right in front of you, reach forward"),
watches their other hand enter the frame, guides it ("move your hand to the right...
almost there"), and announces the grasp when the fingertip lands on the cup.

No special hardware. No app install. Just a phone browser, a WebSocket, and a Python
backend doing the vision and speech work.

## Highlights

- **Voice in, voice out.** Push-to-talk recording, server-side faster-whisper for STT,
  gTTS for synthesis. No screen interaction required.
- **Three task families.** Find objects in a room, navigate room-to-room by exploring
  door-by-door, and reach for an object within arm's reach.
- **YOLOv8 + MediaPipe Hands.** Object detection and 21-landmark hand pose, both running
  on CPU, ~5 FPS end-to-end.
- **Custom-trained door detector — mAP@50 0.946.** Fills the one gap COCO leaves
  (no door class); see [the door detector](#the-trained-door-detector).
- **Per-class distance estimation.** A laptop at 40% of frame width is "near"; a cup at
  18% is *also* near — same camera, correct guidance per object.
- **Auto-grasp completion.** When the fingertip enters the target's box, the task ends
  automatically and announces success.
- **iOS-friendly audio.** Persistent primed `<audio>` element + Web Audio unlock so TTS
  actually plays on iPhone Safari and Chrome.
- **Phone-deployable in minutes.** Same-origin frontend + Cloudflare quick tunnel = a
  real HTTPS URL the phone can hit, no certificates to manage.
- **280+ tests** over the pure decision logic, running in under a second.

---

## The trained door detector

The navigation task needs doorways as targets, but COCO has no `door` class — so a
single-class detector was trained from scratch (transfer-learned from COCO weights).

**Datasets:** DoorDetect (386 doors + 827 hard negatives) merged with DeepDoors2 (3,000
images, masks converted to boxes) → **4,213 images, one `door` class, 3,371 / 842
train-val split**.

**Setup:** YOLOv8, imgsz 640, batch 8, AdamW, AMP; best checkpoint at epoch 39.

| Metric | Value |
|:--|:--:|
| **mAP@50** | **0.946** |
| **mAP@50-95** | **0.829** |
| **Precision** | **0.968** |
| **Recall** | **0.892** |

The 0.97 precision (very few false doors) is helped by the hard-negative images;
this comfortably clears the project's ≥0.60 mAP@50 target. Full write-up in
[`docs/Door_Model_Training_Results.md`](docs/Door_Model_Training_Results.md).

At runtime the door detector is one layer of a **three-stage funnel**: the trained
detector → geometric gates (aspect ratio, edge density, frame-fill) → a 4-class
DoorDetect verifier as a semantic second opinion (kills curtains and door↔fridge
confusions).

---

## Quickstart

### Requirements

- Python 3.13 (3.11+ works; 3.13 is what's verified).
- A modern browser (tested: Chrome on Android, Safari on iOS, Chrome on desktop).
- Optional but recommended for phone testing:
  [`cloudflared`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/).

### Run the backend

```bash
cd backend
python -m venv .venv

# Activate
source .venv/Scripts/activate          # Git Bash on Windows
source .venv/bin/activate               # macOS / Linux
.\.venv\Scripts\Activate.ps1            # Windows PowerShell

pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

First request triggers a few one-time downloads:

- Whisper `base` weights (~150 MB, HuggingFace, cached in `~/.cache/huggingface/`).
- YOLOv8n weights (`yolov8n.pt`, ~6 MB, ultralytics CDN).
- MediaPipe Hands models ship inside the pip wheel — no download.
- First *navigation* task additionally pulls YOLOv8m (~50 MB) and SegFormer-B0 (~15 MB).
  The custom door detector + verifier weights are committed (`best.pt`,
  `door_training/runs/...`) — no download.

### Use it

Open <http://localhost:8000> in any browser. The frontend is served by FastAPI itself,
so the page and the WebSocket share a single origin. Press **Start**, hold **PTT**, say
"find my cup."

### Phone-test it (Cloudflare Tunnel)

Browsers require HTTPS for camera + microphone access except on `localhost`:

```bash
# In a second terminal (keep uvicorn running)
cloudflared tunnel --url http://localhost:8000
```

Open the printed `https://<random>.trycloudflare.com` URL on your phone. No cert
provisioning, no router config.

---

## Supported commands

| User says | Result |
| --- | --- |
| "find my **cup**" / "where is my **bottle**" / "look for the **laptop**" | Start an Object Allocation task. |
| "navigate to the **kitchen**" / "take me to the **bathroom**" | Start a Navigation task. |
| "**got it**" / "found it" / "thanks" / "done" | Mark the current task complete, return to listening. |
| "**stop**" / "cancel" / "never mind" | Abort the current task; the session stays alive. |

Object Allocation supports a curated set of COCO classes (cup, bottle, chair, couch, bed,
dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven,
sink, refrigerator, book, clock, vase, scissors). Common mishearings ("phone" → "cell
phone", "fridge" → "refrigerator") are normalised before fuzzy matching.

---

## Architecture

```
+---------------------+                +-----------------------------------+
|  Browser (phone)    |    WebSocket   |  FastAPI backend                  |
|                     |  <---------->  |  /ws (single connection per user) |
|  - getUserMedia     |                |                                   |
|  - MediaRecorder    |   JSON +       |  Router -> Session -> FSM         |
|  - <audio> element  |   binary tags  |                                   |
|  - Push-to-talk UI  |                |  Services:                        |
|  - compass heading  |                |    faster-whisper  (STT)          |
+---------------------+                |    YOLOv8n         (detection)    |
                                       |    MediaPipe Hands (hand pose)    |
                                       |    spatial_reasoning + guidance   |
                                       |    navigation (door model +       |
                                       |      exploration + obstacles)     |
                                       |    gTTS            (synthesis)    |
                                       +-----------------------------------+
```

A single `/ws` connection carries JSON control messages and three tagged binary frames:
`0x01` JPEG camera frame (client→server, 5 FPS), `0x02` WebM/Opus PTT audio (client→server,
on PTT release), `0x03` MP3 TTS clip (server→client). See
[`docs/protocol.md`](docs/protocol.md) for the frozen wire contract.

The backend is a single uvicorn process. Each connected user gets a `Session` that owns
its own FSM, latest decoded frame, task context, and detection-loop asyncio task. There
is no shared task state across users.

### Task FSM

```
                  user_start
                ----------------->
   Idle                              ListeningForCommand
   ^                                    |
   |              cleanup_done          | command_recognized
   |              <-------              v
   ReturningToIdle <-----+    ObjectAllocationActive  /  NavigationActive
            ^             \              |
            |              \   user_stop |                task_abort
            |               \            v               ------------>
            +---- task_complete <--- (active states) ----> back to LISTENING
```

The FSM is authoritative — the client just renders whatever `fsm_state` the server pushes.
Every transition is driven by an explicit event; the only autonomous transitions are
`cleanup_done` and `task_complete` on grasp.

---

## Task families

### Object Allocation — find a thing in the room

Per frame, 5 FPS:

1. **Detect** the requested COCO class with YOLOv8n; drop boxes below 0.35 confidence.
2. **Temporal filter** — the target must appear in 3 of the last 5 frames before it's
   trusted. Kills single-frame flickers.
3. **Spatial reasoning** — region (left / center / right, split at 0.35 / 0.65 of width)
   and distance (near / medium / far) via `max(width_frac, height_frac)` against a
   per-COCO-class threshold.
4. **Speak** a throttled, region-aware phrase, re-announced only when the bucket changes
   or 6 s elapse.

Edge cases handled: periodic scan prompt if never seen in 30 s; one-shot "I lost sight of
your cup" on loss; auto-abort at 60 s; multi-instance → guide to the most head-on one.

### Reach Guidance — hand-relative cues

When the target is at `near` distance AND MediaPipe detects a hand, the loop switches to
hand-relative cues: `approach` ("move your hand to the right"), `almost` (fingertip within
10% of the target centroid → "almost there, reach forward"), and `touching` (fingertip
inside the box → "your hand is on the cup, grasp it"). The `touching` state is the **only
autonomous success-exit** — it fires `task_complete` automatically. MediaPipe is only
invoked near reach distance, saving CPU.

### Navigation — goal-directed exploration

The user names only a destination and Lumen guides them there room-by-room. Per room:
**scan** (compass-tracked 360°, detections bucketed into 30° sectors) → **arrive?**
(rooms recognised by their objects, temporally accumulated) → **choose a door** (the
three-stage funnel above) → **go** (bearing + step count from pinhole distance, with an
obstacle watchdog fusing YOLO objects and SegFormer floor segmentation) → **cross**
(door box saturates then vanishes *plus* sustained camera motion — the camera is the
odometer). Runs at ~3 Hz. See
[`docs/Exploration_Navigation_Design.md`](docs/Exploration_Navigation_Design.md).

---

## Project structure

```
Lumen/
├── backend/                       # FastAPI server (single process)
│   ├── main.py                    # /health, /ws, mounts ../frontend
│   ├── api/                       # session, router, frame/audio handlers
│   ├── fsm/task_fsm.py            # 5 states, explicit transitions only
│   ├── services/
│   │   ├── speech_service.py      # faster-whisper + PyAV (no ffmpeg needed)
│   │   ├── command_parser.py      # rapidfuzz intent extraction
│   │   ├── tts_service.py         # gTTS + bounded LRU cache
│   │   ├── yolo_service.py        # YOLOv8n via ultralytics
│   │   ├── hand_service.py        # MediaPipe Hands landmark->pose
│   │   ├── spatial_reasoning.py   # per-class distance + region classifier
│   │   ├── guidance_generator.py  # Object Allocation phrase templates
│   │   ├── reach_guidance.py      # fingertip-vs-target logic + phrases
│   │   ├── object_allocation.py   # GuidanceTracker + 5 Hz async loop
│   │   └── navigation/            # goal-directed exploration
│   │       ├── engine.py / controller.py / perception.py / obstacles.py
│   │       ├── geometry.py / goals.py / models.py / state.py / config.py
│   └── tests/                     # 280+ pytest cases
├── frontend/                      # Vanilla HTML + JS, no build step
│   └── js/                        # app, ws_client, media, audio_queue, heading, wakelock
├── door_training/                 # door-model dataset prep + trained verifier
├── best.pt                        # custom single-class door detector (committed)
├── docs/                          # protocol, navigation design, door results, plan
└── README.md
```

---

## Development

### Running tests

```bash
cd backend
python -m pytest tests/ -q
```

280+ tests cover FSM transitions, command parsing (50+ realistic transcriptions with
mishearings and synonyms), spatial bucketing, guidance-phrase rendering, the
reach-guidance state machine, hand-pose conversion, YOLO parsing, the Object Allocation
tracker end-to-end (scripted frames + fake clock), and the navigation controller
end-to-end plus its compass/bearing math. The suite mocks the heavy ML models, so it runs
in under a second.

### Tech stack

| Layer | Choice | Why |
| --- | --- | --- |
| Server | FastAPI + uvicorn | Async WebSockets, fast iteration, type hints. |
| STT | faster-whisper (`base`) | CTranslate2 backend; 4× faster CPU than openai-whisper; no torch dep. |
| Audio decode | PyAV | Bundles FFmpeg in the wheel — no system `ffmpeg` on PATH. |
| Object detection | YOLOv8n | Smallest of the family (~6 MB), CPU-friendly, COCO matches the noun list. |
| Hand pose | MediaPipe Hands | 21 landmarks, CPU realtime, models bundled in wheel. |
| Navigation detection | YOLOv8m + custom door YOLOv8 ×2 | v8m for room indicators; trained single-class door detector + 4-class verifier. |
| Obstacle floor check | SegFormer-B0 (ADE20K) | Per-pixel labels; walking lane must stay mostly floor. Degrades to YOLO-only. |
| TTS | gTTS | Free; MP3s cached; latency masked by parallel detection. |
| Command parsing | rapidfuzz | Tolerant of Whisper mishearings. |
| Frontend | Vanilla HTML/JS, no build | One less moving part; served same-origin by the backend. |

---

## Roadmap

- [x] Sprint 1 — voice round-trip end-to-end (FSM + WS + STT + TTS).
- [x] Sprint 2 — phone deployment over HTTPS (Cloudflare Tunnel + iOS audio unlock).
- [x] Sprint 3 — Object Allocation with YOLOv8n.
- [x] Sprint 4 — Navigation via goal-directed exploration (custom door model + room recognition + obstacle watchdog).
- [x] Sprint 5 — Reach Guidance with MediaPipe Hands.
- [ ] Sprint 6 — Blindfolded user trials, performance polish, final report.

See [`docs/Lumen_Implementation_Plan.pdf`](docs/Lumen_Implementation_Plan.pdf) for the full plan.

---

## Author

- **Diaa Badaha** — 1210478 — Birzeit University, ENCS5200.

Developed as a graduation project at Birzeit University's Department of Electrical and
Computer Engineering — covering the backend architecture, the vision and reach-guidance
pipeline, and the exploration-navigation pipeline (door-model training, perception funnel,
and exploration controller).

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
