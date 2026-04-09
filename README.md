# Footstep Tracker

![Build Status](https://github.com/manorajesh/footsteps/actions/workflows/rust.yml/badge.svg)
![Downloads](https://img.shields.io/crates/d/footstep-tracker)
![Version](https://img.shields.io/crates/v/footstep-tracker)
![License](https://img.shields.io/crates/l/footstep-tracker)
![Size](https://img.shields.io/crates/size/footstep-tracker)

<div align="center">
  <img src="https://github.com/manorajesh/footsteps/blob/8b07a8251b91a4464efb1c49fd9ba6742838aa2a/images/screenshot.png?raw=true" alt="Footstep Tracker Screenshot" width="80%">
  <video src="https://github.com/manorajesh/footsteps/blob/3a13f994a0762db1a674eafa8dcc73b295dedded/images/demo_video.mp4?raw=true" width="80%" controls></video>
</div>

Real-time multi-person footstep detection on macOS using CoreML (YOLOv11 for person boxes + RTMPose for 17-keypoint pose) with OpenCV and Rust.

## What it does

- Runs fully on-device with CoreML (CPU + Apple Neural Engine when available)
- Detects people with a YOLOv11 CoreML model (`models/yolo11n.mlpackage`)
- estimation of 17 COCO keypoints per person with an RTMPose SimCC CoreML model (`models/rtmpose.mlpackage`)
- Tracks per-person IDs and emits footsteps when ankles stop after moving
- Optional UDP or OSC output of footsteps for downstream consumers (e.g., Unity/TouchDesigner)
- Works with webcam or video files; video sources loop automatically

## Requirements (macOS)

- Rust toolchain: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- OpenCV: `brew install opencv`
- LLVM (if not already installed with Xcode): `brew install llvm`
- macOS with CoreML available (runs best on Apple Silicon); allow camera access for the terminal/IDE

## Models

- Provided: `models/rtmpose.mlpackage` (256x192 RTMPose-SimCC, 17 keypoints)
- Provided: `models/yolo11n.mlpackage` (person detector)
- You can swap in your own CoreML `.mlpackage`/`.mlmodelc` files with matching inputs; place them in `models/` and point the CLI at the path.

## Build

```bash
  git clone --recurse-submodules https://github.com/manorajesh/footsteps.git
  ```

## Build

- Recommended (backend + frontend together):

  ```bash
  python3 run_dev.py
  ```

- With debug feature enabled for backend logs/timings:

  ```bash
  python3 run_dev.py --debug
  ```

- With a specific backend source (camera index or video file):

  ```bash
  # camera 1
  python3 run_dev.py --video 1

  # mp4 file
  python3 run_dev.py --video people.mp4

  # combine with debug
  python3 run_dev.py --debug --video people.mp4
  ```

- Runner behavior:

  ```text
  - Creates/uses ./venv automatically
  - Installs Python deps from footsteps_projection_mapping/requirements.txt
  - Starts backend and frontend with unified logging
  - Restarts crashed processes automatically
  ```

- Backend-only release build (recommended for FPS):

  ```bash
  cargo build --release
  ```

- Backend-only dev build:

  ```bash
  cargo build
  ```

- Backend-only debug feature build:

  ```bash
  cargo build --release --features debug
  ```

## Run

- Recommended app run (backend + frontend):

  ```bash
  python3 run_dev.py
  ```

- Debug runner mode:

  ```bash
  python3 run_dev.py --debug
  ```

- Runner with camera/video source forwarded to backend:

  ```bash
  # camera 0
  python3 run_dev.py --video 0

  # video file
  python3 run_dev.py --video /path/to/video.mp4
  ```

- Backend-only webcam defaults (RTMPose + YOLO, camera 0):

  ```bash
  cargo run --release
  ```

- Backend-only: specify model and source (camera index or video path):

  ```bash
  # camera 1
  cargo run --release -- -m models/rtmpose.mlpackage 1

  # mp4 file (auto-loops)
  cargo run --release -- -m models/rtmpose.mlpackage /path/to/video.mp4
  ```

- OSC output for footsteps (sends `/footstep` with data over UDP):

  ```bash
  cargo run --release -- -m models/rtmpose.mlpackage -o 192.168.1.42:7001
  ```

- UDP output for footsteps (sends space-separated text payload):

  ```bash
  cargo run --release -- -m models/rtmpose.mlpackage -u 192.168.1.42:7000
  ```

- Controls: press `q` to quit.

## How it works

1. Capture frames from AVFoundation (webcam) or video file
2. Run YOLOv11 CoreML to get person boxes (normalized)
3. Track stable person IDs across frames
4. Crop each person box, resize to 256x192, run RTMPose SimCC CoreML to get 17 keypoints
5. Map keypoints back to full-frame coordinates
6. Detect footsteps when ankles transition from moving to still; keep recent trails and archive past visitors
7. Draw IDs, boxes, and footsteps (color-coded per person) and optionally emit UDP events

## Coordinates and Payload Formats

- Keypoints and footsteps are normalized to the frame: `x` and `y` are in `[0,1]`.

**UDP:** 
Payload per footstep is sent as a `\n` terminated text string.
Format: `x y person_id history_length hx_1 hy_1 hx_2 hy_2 ...`
Example: `0.4123 0.7831 2 1 0.4100 0.8100`

**OSC:**
Payload per footstep is sent via `/footstep` message with matching format sequence as arguments:
`[float(x), float(y), int(person_id), int(history_length), float(hx_1), float(hy_1), ...]`

## Project structure

```
src/
  main.rs            - CLI/entrypoint, capture loop, UDP hook
  person_detector.rs - YOLOv11 CoreML person detector + ID tracker
  pose_detector.rs   - RTMPose SimCC CoreML inference
  footstep_tracker.rs- Ankle-motion based footstep detection & history
  visualization.rs   - Drawing boxes, footsteps, and (optional) keypoints

coreml-rs/           - Local CoreML bindings (Swift + Rust bridge)
models/              - CoreML model packages (YOLO + RTMPose provided)
```

## Troubleshooting

- OpenCV not found: `brew reinstall opencv` and ensure Homebrew is on your PATH (`eval "$($(brew --prefix)/bin/brew shellenv)"`).
- Webcam fails: check macOS camera permissions and try another index (`cargo run --release -- models/rtmpose.mlpackage 1`).
- CoreML load error: verify the `.mlpackage`/`.mlmodelc` path exists and you are on macOS with CoreML available.
- Low FPS: prefer `--release`, close other GPU/NE heavy apps, and keep the preview window small.

## License

Model licenses follow their respective sources (YOLOv11 and RTMPose). Code is under the repository license.
