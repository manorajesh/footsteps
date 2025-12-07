# Footstep Tracker

![Build Status](https://github.com/manorajesh/footsteps/actions/workflows/rust.yml/badge.svg)

<div align="center">
  <img src="https://github.com/manorajesh/footsteps/blob/8b07a8251b91a4464efb1c49fd9ba6742838aa2a/images/screenshot.png?raw=true" alt="Footstep Tracker Screenshot" width="80%">
</div>

Real-time multi-person footstep detection on macOS using CoreML (YOLOv11 for person boxes + RTMPose for 17-keypoint pose) with OpenCV and Rust.

## What it does

- Runs fully on-device with CoreML (CPU + Apple Neural Engine when available)
- Detects people with a YOLOv11 CoreML model (`models/yolo11n.mlpackage`)
- Estimates 17 COCO keypoints per person with an RTMPose SimCC CoreML model (`models/rtmpose.mlpackage`)
- Tracks per-person IDs and emits footsteps when ankles stop after moving
- Optional UDP output of footsteps for downstream consumers (e.g., Unity/TouchDesigner)
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

- Release (recommended for FPS):

  ```bash
  cargo build --release
  ```

- Dev build:

  ```bash
  cargo build
  ```

- Debug logs (timings, model info):

  ```bash
  cargo build --release --features debug
  ```

## Run

- Webcam with defaults (RTMPose + YOLO, camera 0):

  ```bash
  cargo run --release
  ```

- Specify model and source (camera index or video path):

  ```bash
  # camera 1
  cargo run --release -- models/rtmpose.mlpackage 1

  # mp4 file (auto-loops)
  cargo run --release -- models/rtmpose.mlpackage /path/to/video.mp4
  ```

- UDP output for footsteps (sends `"<x> <y>"` normalized to 0-1):

  ```bash
  FOOTSTEP_UDP_ADDR=192.168.1.42:5005 cargo run --release -- models/rtmpose.mlpackage
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

## Coordinates

- Keypoints and footsteps are normalized to the frame: `x` and `y` are in `[0,1]`.
- UDP payload per footstep: `<x> <y>` on a single line (e.g., `0.4123 0.7831`).

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
