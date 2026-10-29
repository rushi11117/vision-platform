# Vision Platform — Complete Documentation

> **Simple version:** This is a plug-and-play computer-vision framework.  
> You point it at a camera, tell it what to detect, and it saves/alerts/displays the results — all configurable from one YAML file.

---

## Table of Contents

1. [What Does It Do?](#1-what-does-it-do)
2. [Folder Structure](#2-folder-structure)
3. [Quick Start](#3-quick-start)
4. [Configuration Reference (config.yaml)](#4-configuration-reference)
5. [Input Sources](#5-input-sources)
6. [Detectors](#6-detectors)
7. [Processors (What Happens After Detection)](#7-processors)
8. [Database / Storage](#8-database--storage)
9. [Multi-Camera Support](#9-multi-camera-support)
10. [Adding a New Detector](#10-adding-a-new-detector)
11. [Adding a New Source](#11-adding-a-new-source)
12. [Running Tests](#12-running-tests)
13. [Error Handling](#13-error-handling)
14. [Architecture & Design Principles](#14-architecture--design-principles)
15. [FAQ](#15-faq)

---

## 1. What Does It Do?

```
Camera / Video / Image
        ↓
   Frame Extraction
        ↓
  Detection Engine  ← runs face, person, motion detectors simultaneously
        ↓
    Processors      ← display on screen, save images, record video, save to DB, send alerts
        ↓
 Storage / Database
```

In plain English:
- **Input**: any camera, video file, folder of images, RTSP stream, or phone camera
- **Detect**: faces, people, motion (more can be added)
- **Output**: show on screen, save snapshots, record video, log to database, email/webhook alerts

---

## 2. Folder Structure

```
vision_platform/
│
├── app.py                    ← START HERE — main entry point
├── requirements.txt          ← Python packages needed
│
├── config/
│   ├── config.yaml           ← ALL settings in one place
│   ├── settings.py           ← loads config.yaml into Python objects
│   └── logging_config.py     ← sets up log files + console output
│
├── core/                     ← abstract interfaces (no concrete code here)
│   ├── contracts/            ← "rules" each module must follow
│   │   ├── frame_source.py   ← rule: all sources must have frames() and release()
│   │   ├── detector.py       ← rule: all detectors must have detect(frame)
│   │   ├── processor.py      ← rule: all processors must have process(frame, detections, meta)
│   │   └── repository.py     ← rule: all DBs must have save() and get_recent()
│   ├── models/
│   │   ├── detection_result.py  ← the data object that flows through the system
│   │   └── frame_metadata.py    ← camera name, timestamp, frame number
│   └── exceptions/
│       └── custom_exceptions.py ← all error types
│
├── sources/                  ← WHERE frames come from
│   ├── image_source.py       ← single image file
│   ├── folder_source.py      ← folder of images
│   ├── video_source.py       ← MP4 / AVI / MOV file
│   ├── webcam_source.py      ← USB or laptop camera
│   ├── rtsp_source.py        ← RTSP CCTV / IP camera / HTTP stream
│   └── android_camera_source.py ← phone camera (DroidCam / IP Webcam app)
│
├── detectors/                ← WHAT to look for
│   ├── face_detector.py      ← Haar Cascade faces
│   ├── person_detector.py    ← HOG+SVM people
│   ├── motion_detector.py    ← frame diff / MOG2
│   └── qr_detector.py        ← QR codes
│
├── engines/
│   ├── detection_engine.py   ← runs all detectors on one frame
│   └── pipeline_engine.py    ← Source → Engine → Processors loop
│
├── processors/               ← WHAT to do with detections
│   ├── display_processor.py  ← show live window
│   ├── image_save_processor.py  ← save snapshot JPGs
│   ├── video_save_processor.py  ← record MP4 video
│   ├── database_processor.py ← write to SQLite / PostgreSQL
│   └── alert_processor.py    ← email / webhook alerts
│
├── repositories/             ← HOW to talk to the database
│   ├── sqlite_repository.py  ← Phase 1 (default)
│   └── postgres_repository.py ← Phase 2
│
├── services/
│   └── stream_service.py     ← manages multiple cameras with threads
│
├── workers/
│   └── camera_worker.py      ← one thread per camera
│
├── utils/
│   ├── image_utils.py        ← draw boxes, resize, encode
│   └── datetime_utils.py     ← timestamp helpers
│
├── tests/
│   ├── test_detectors.py
│   ├── test_sources.py
│   └── test_pipeline.py
│
└── storage/
    ├── images/               ← saved detection images
    ├── videos/               ← recorded videos
    ├── snapshots/            ← auto-named snapshots
    └── logs/                 ← platform.log
```

---

## 3. Quick Start

### Install

```bash
pip install -r requirements.txt
```

> **Note**: `opencv-python==4.5.4.60` is locked. Do not upgrade.

### Run on your webcam (default)

```bash
python app.py
```

Press **q** to quit.

### Run on a video file

```bash
python app.py --video path/to/video.mp4
```

### Run on a single image

```bash
python app.py --image path/to/photo.jpg
```

### Run on an RTSP stream

```bash
python app.py --rtsp rtsp://admin:pass@192.168.1.100:554/live
```

### Run with custom config

```bash
python app.py --config config/my_custom.yaml
```

---

## 4. Configuration Reference

Everything is controlled by **`config/config.yaml`**. You never need to touch Python code for normal use.

### 4.1 Cameras

```yaml
cameras:
  - name: Gate1              # label shown on screen and saved in DB
    stream: 0                # can be: webcam index, file path, rtsp:// URL, http:// URL
    enabled: true            # set false to temporarily disable without deleting
    detectors: [face, person, motion]  # which detectors run on this camera
```

**Stream values and what they mean:**

| Value | What it does |
|-------|-------------|
| `0`, `1`, `2` | Laptop/USB webcam by index |
| `./video.mp4` | Video file |
| `./images/` | Folder of images |
| `rtsp://...` | RTSP CCTV camera |
| `http://192.168.x.x:8080/video` | Android phone (IP Webcam app) |

### 4.2 Detection Settings

```yaml
detection:
  face:
    scale_factor: 1.1     # how aggressively to downscale (lower = slower but finds more)
    min_neighbors: 5      # higher = fewer false positives
    min_size: [30, 30]    # ignore faces smaller than this (pixels)

  person:
    confidence_threshold: 0.5   # ignore detections below this score

  motion:
    threshold: 1000       # minimum moving area in pixels² to count as motion
    background_subtractor: false  # true = smarter MOG2, false = simple frame diff
```

### 4.3 Processors

```yaml
processors:
  display:
    enabled: true          # show live window
    draw_boxes: true       # draw rectangles around detections

  image_save:
    enabled: true
    output_dir: ./storage/snapshots
    save_on_detection: true   # false = save every frame

  video_save:
    enabled: false
    mode: event            # "continuous" = always record, "event" = only on detection

  database:
    enabled: true          # save every detection to SQLite

  alert:
    enabled: false
    cooldown_seconds: 60   # wait 60s before sending same alert again
    email:
      enabled: false
      smtp_host: smtp.gmail.com
      sender: you@gmail.com
      recipients: [boss@company.com]
```

### 4.4 Database

```yaml
database:
  type: sqlite              # switch to "postgres" for Phase 2
  db_path: ./database/vision.db
```

### 4.5 Multi-Camera Workers

```yaml
workers:
  max_workers: 10           # max parallel camera threads
  frame_skip: 0             # 0 = process every frame, 2 = every 3rd frame (faster)
  reconnect_delay_seconds: 5
  max_reconnect_attempts: 10
```

---

## 5. Input Sources

Each source inherits from `FrameSource` and provides a `frames()` generator.

### ImageSource
- **File**: `sources/image_source.py`
- **What**: Reads one image file, yields one frame
- **Use when**: Testing with a photo, or processing a single image

### FolderSource
- **File**: `sources/folder_source.py`
- **What**: Reads every image in a folder (sorted alphabetically), yields them as frames
- **Supported formats**: jpg, jpeg, png, bmp, tiff, webp

### VideoSource
- **File**: `sources/video_source.py`
- **What**: Reads a video file frame by frame
- **Supported formats**: mp4, avi, mov, mkv, flv, wmv
- **Tip**: Set `frame_skip: 2` to process every 3rd frame for speed

### WebcamSource
- **File**: `sources/webcam_source.py`
- **What**: Streams from USB or built-in camera
- **Auto-reconnect**: if camera disconnects, waits and retries N times

### RTSPSource
- **File**: `sources/rtsp_source.py`
- **What**: Streams from any RTSP URL, IP camera, or HTTP video stream
- **Auto-reconnect**: handles network drops automatically

### AndroidCameraSource
- **File**: `sources/android_camera_source.py`
- **What**: Thin wrapper for phone camera apps
- **Compatible apps**: IP Webcam (port 8080), DroidCam (port 4747)
- **Usage**: Install app on phone → connect to same WiFi → use phone IP in config

---

## 6. Detectors

Each detector inherits from `Detector` and implements `detect(frame) → List[DetectionResult]`.

### FaceDetector
- **Algorithm**: OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`)
- **Output label**: `"face"`
- **Best for**: Close-up frontal faces
- **Config**: `detection.face` in config.yaml

### PersonDetector
- **Algorithm**: HOG (Histogram of Oriented Gradients) + SVM
- **Output label**: `"person"`
- **Best for**: Full-body people detection from a distance
- **Config**: `detection.person` in config.yaml

### MotionDetector
- **Algorithm**: Frame differencing (default) or MOG2 background subtraction
- **Output label**: `"motion"`
- **Two modes**:
  - `background_subtractor: false` → compare consecutive frames (simple, fast)
  - `background_subtractor: true` → MOG2 model (smarter, handles lighting changes)
- **Config**: `detection.motion` in config.yaml

### QRDetector
- **Algorithm**: OpenCV QRCodeDetector
- **Output label**: `"qr:<decoded_text>"`
- **Output**: Also decodes the QR text content

### DetectionResult — the common data object

Every detector returns a list of `DetectionResult` objects:

```python
DetectionResult(
    camera_name = "Gate1",
    detector    = "FaceDetector",
    label       = "face",
    confidence  = 0.95,
    timestamp   = datetime(2026, 10, 21, 10, 0, 0),
    bounding_box = (x, y, width, height),   # in pixels
    frame_index = 42,
)
```

---

## 7. Processors

Processors receive every frame + its detections. They run in the order defined.

### DisplayProcessor
- **What**: Opens an OpenCV window and shows live annotated video
- **Controls**: Press `q` to stop the pipeline
- **Shows**: Coloured bounding boxes + labels on screen
  - Green = face, Blue-Orange = person, Red = motion

### ImageSaveProcessor
- **What**: Saves frames as JPEG/PNG snapshots
- **When**: Only when `save_on_detection: true` AND detections exist
- **Filename**: `20261021_103000_face.jpg` (timestamp + label)
- **Location**: `storage/snapshots/`
- **Disk guard**: Warns and stops if disk has less than 100 MB free

### VideoSaveProcessor
- **What**: Records processed frames to MP4 video
- **Two modes**:
  - `continuous` — always recording, one file per session
  - `event` — starts recording when detection happens, stops after 5s idle
- **Pre-buffer**: In event mode, saves a few seconds before the detection (configurable)

### DatabaseProcessor
- **What**: Saves every `DetectionResult` to the database
- **Thread-safe**: Multiple cameras can write simultaneously
- **Columns stored**: camera_name, detector, label, confidence, timestamp, bounding_box, frame_index

### AlertProcessor
- **What**: Sends notifications when detections occur
- **Channels**: Email (SMTP), Webhook (HTTP POST), SMS\*, WhatsApp\*
- **Cooldown**: Won't spam — waits N seconds between same-type alerts
- **Non-blocking**: Alerts fire in background threads so they don't slow down detection

\*SMS/WhatsApp stubs ready to implement with Twilio or similar

---

## 8. Database / Storage

### SQLite (Phase 1 — Default)

- No setup needed — file created automatically at `./database/vision.db`
- WAL mode enabled for concurrent multi-camera writes
- Thread-safe via `threading.Lock`

### PostgreSQL (Phase 2)

1. Install: `pip install psycopg2-binary`
2. Change in config.yaml:
   ```yaml
   database:
     type: postgres
     host: localhost
     port: 5432
     name: vision_db
     user: vision_user
     password: ""   # or export DB_PASSWORD=yourpassword
   ```

### Database Schema

```sql
CREATE TABLE detections (
    id          INTEGER PRIMARY KEY,
    camera_name TEXT,
    detector    TEXT,
    label       TEXT,
    confidence  REAL,
    timestamp   TEXT,
    bbox_x      INTEGER,
    bbox_y      INTEGER,
    bbox_w      INTEGER,
    bbox_h      INTEGER,
    frame_index INTEGER
);
```

### Storage Layout

```
storage/
├── images/      ← general images (future use)
├── videos/      ← recorded video files
├── snapshots/   ← auto-saved detection snapshots
└── logs/
    └── platform.log   ← rotating log (max 10 MB × 5 files)
```

---

## 9. Multi-Camera Support

The platform supports 1 to 50+ cameras simultaneously.

**How it works:**

```
StreamService
    │
    ├── ThreadPoolExecutor (max_workers = 10 by default)
    │       │
    │       ├── CameraWorker (Gate1)  → VisionPipeline → runs in thread 1
    │       ├── CameraWorker (Gate2)  → VisionPipeline → runs in thread 2
    │       ├── CameraWorker (Lobby)  → VisionPipeline → runs in thread 3
    │       └── ...
    │
    └── One shared SQLiteRepository (thread-safe)
```

**To add more cameras**, just add entries in config.yaml:

```yaml
cameras:
  - name: Gate1
    stream: rtsp://192.168.1.100/live
    enabled: true
    detectors: [person, motion]

  - name: Lobby
    stream: rtsp://192.168.1.101/live
    enabled: true
    detectors: [face, person]

  - name: Parking
    stream: 0
    enabled: true
    detectors: [motion]
```

**Resilience**: If one camera crashes, only that worker stops. All other cameras keep running.

---

## 10. Adding a New Detector

**Example: Adding a `FireDetector`**

**Step 1**: Create `detectors/fire_detector.py`:

```python
import cv2
import numpy as np
from core.contracts.detector import Detector
from core.models.detection_result import DetectionResult

class FireDetector(Detector):
    @property
    def detector_name(self):
        return "FireDetector"

    def detect(self, frame):
        # your detection logic here
        # return [] if no fire, else list of DetectionResult
        return []
```

**Step 2**: Register in `engines/detection_engine.py`:

```python
from detectors.fire_detector import FireDetector

DETECTOR_MAP = {
    "face":   FaceDetector,
    "person": PersonDetector,
    "motion": MotionDetector,
    "fire":   FireDetector,   # ← add this line
}
```

**Step 3**: Enable in config.yaml:

```yaml
cameras:
  - name: KitchenCam
    stream: rtsp://...
    detectors: [motion, fire]   # ← add "fire"
```

**That's it. No other files need to change.**

---

## 11. Adding a New Source

**Example: Adding a `GStreamerSource`**

```python
# sources/gstreamer_source.py
from core.contracts.frame_source import FrameSource

class GStreamerSource(FrameSource):
    def __init__(self, pipeline_str: str, name: str = "GStreamer"):
        self._pipeline = pipeline_str
        self._name = name

    @property
    def name(self):
        return self._name

    def frames(self):
        import cv2
        cap = cv2.VideoCapture(self._pipeline, cv2.CAP_GSTREAMER)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()

    def release(self):
        pass
```

Then use it directly:

```python
from sources.gstreamer_source import GStreamerSource
source = GStreamerSource("v4l2src device=/dev/video0 ! videoconvert ! appsink")
```

Or add it to `CameraWorker._build_source()` for config-driven use.

---

## 12. Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_detectors.py -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=. --cov-report=term-missing
```

### What's tested

| Test file | Tests |
|-----------|-------|
| `test_detectors.py` | FaceDetector, PersonDetector, MotionDetector: blank frames, noisy frames, config options |
| `test_sources.py` | ImageSource, FolderSource, VideoSource: loading, skipping, error handling |
| `test_pipeline.py` | DetectionEngine + VisionPipeline: integration, metadata flow, DB save/retrieve |

---

## 13. Error Handling

The platform is designed to **never crash completely** due to one bad camera or file.

| Situation | What happens |
|-----------|-------------|
| Camera disconnects | Worker retries N times with delay, then logs and stops only that camera |
| RTSP stream times out | Same reconnect logic as above |
| Corrupt video file | `CorruptVideoError` logged, pipeline stops gracefully |
| Invalid image | `InvalidImageError` logged, source returns zero frames |
| Detector fails | Error logged, other detectors on same frame still run |
| Database error | Logged, frame processing continues |
| Disk full | `DiskFullError` raised, image/video saving stops |
| Other cameras | All other cameras keep running regardless |

### Error hierarchy

```
VisionPlatformError
├── SourceError
│   ├── CameraDisconnectedError
│   ├── RTSPTimeoutError
│   ├── CorruptVideoError
│   └── InvalidImageError
├── DetectorError
│   └── ModelLoadError
├── ProcessorError
│   └── DiskFullError
├── RepositoryError
│   └── DatabaseConnectionError
└── AlertError
```

---

## 14. Architecture & Design Principles

### SOLID Principles in Practice

| Principle | How it's implemented |
|-----------|---------------------|
| **S** ingle Responsibility | `RTSPSource` only reads frames. `FaceDetector` only finds faces. `SQLiteRepository` only handles DB. Each class does one thing. |
| **O** pen/Closed | Add `FireDetector` without touching any existing file. Register in `DETECTOR_MAP` only. |
| **L** iskov Substitution | Any `Detector` subclass can replace another. `DetectionEngine` doesn't care which detector it has. |
| **I** nterface Segregation | Small, focused interfaces: `FrameSource` (2 methods), `Detector` (1 method), `Processor` (1 method), `Repository` (3 methods) |
| **D** ependency Inversion | `DatabaseProcessor` depends on abstract `Repository`, not on `SQLiteRepository` directly. Swap SQLite ↔ Postgres in config only. |

### Data Flow

```
config.yaml
    ↓ loaded by settings.py
PlatformConfig
    ↓ read by StreamService
CameraWorker (per camera)
    ↓ builds
  FrameSource → yields np.ndarray frames
  DetectionEngine → runs Detectors → List[DetectionResult]
  Processors → DisplayProcessor, ImageSaveProcessor, DatabaseProcessor, ...
    ↓
  VisionPipeline.run() → loops: frame → detect → process
```

### Thread model

```
Main thread: StreamService.start()
    │
    ├── Thread 1: CameraWorker(Gate1).run()  [VisionPipeline loop]
    ├── Thread 2: CameraWorker(Gate2).run()  [VisionPipeline loop]
    └── Thread 3: AlertProcessor email send  [daemon thread, per alert]
```

---

## 15. FAQ

**Q: How do I add my phone as a camera?**  
A: Install "IP Webcam" app on Android. Start the server in the app. In config.yaml set `stream: http://<your-phone-ip>:8080/video`.

**Q: How do I run headless (no display window)?**  
A: Set `processors.display.enabled: false` in config.yaml. Useful for servers.

**Q: The motion detector triggers constantly on windy trees. How do I fix it?**  
A: Increase `detection.motion.threshold` (e.g. 5000). Or switch to `background_subtractor: true` which handles slow environmental changes better.

**Q: How do I switch from SQLite to PostgreSQL?**  
A: Change `database.type: postgres` in config.yaml and fill in host/port/name/user/password. Run `pip install psycopg2-binary`.

**Q: Can I run only motion detection on one camera and only faces on another?**  
A: Yes. Each camera entry in config.yaml has its own `detectors` list.

**Q: Where do alerts go?**  
A: Email alerts require SMTP config in config.yaml. Webhook sends an HTTP POST with the DetectionResult JSON to any URL you specify.

**Q: How do I set the email password safely?**  
A: Use the environment variable: `export ALERT_EMAIL_PASSWORD=yourpassword`. Don't put it in config.yaml.

**Q: I only want to save images when a person is detected, not on motion.**  
A: The `ImageSaveProcessor` saves when ANY detection occurs. To filter, you can subclass it and check `detections[0].label == "person"` before saving.

**Q: The pipeline is slow on a Raspberry Pi. How do I speed it up?**  
A: Set `workers.frame_skip: 2` to process every 3rd frame. Also increase `detection.motion.threshold` and reduce camera resolution in config.

---

*Vision Platform — built with OpenCV 4.5.4.60, Python 3.9+*
