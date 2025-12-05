# Footstep Tracker

A real-time footstep tracking application using MoveNet pose estimation with CoreML acceleration, written in Rust with OpenCV bindings.

## Features

- Real-time multi-person pose estimation using MoveNet Multipose model
- Hardware acceleration via Apple's CoreML (Neural Engine + GPU)
- Webcam capture and processing with AVFoundation backend
- Visual tracking of all body keypoints and skeleton connections
- Temporal smoothing for stable keypoint tracking
- Optimized for performance on Apple Silicon Macs
- Written in Rust for safety and performance

## Prerequisites

### macOS (Apple Silicon or Intel)

Required dependencies:

- **Rust**: Install from [rustup.rs](https://rustup.rs/)

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

- **OpenCV**: Install via Homebrew
  ```bash
  brew install opencv
  ```

### MoveNet Models

The project uses MoveNet Multipose model in CoreML format. You can either:

1. **Use the included model**: The repository includes a pre-converted `movenet_multipose.mlpackage`

2. **Convert from ONNX** (optional): Use the provided `convert_coreml.py` script:
   ```bash
   pip install coremltools onnx
   python convert_coreml.py
   ```

## Building the Project

### Quick Build (Release Mode)

```bash
cargo build --release
```

This will:

- Compile with full optimizations (`opt-level = 3`)
- Enable Link-Time Optimization (LTO)
- Create binary at `target/release/footstep-tracker`

### Development Build

```bash
cargo build
```

Faster compilation for development, with debug symbols.

### Build with Debug Logging

Enable verbose logging for debugging:

```bash
cargo build --release --features debug
```

This enables:

- Model initialization messages
- CoreML status information
- Frame processing statistics

## Running

### Run with Default Settings

```bash
cargo run --release
```

Uses default model (`models/movenet_multipose.mlpackage`) and camera 0.

### Specify Model and Camera

```bash
# Specify model path
cargo run --release -- models/movenet_multipose.mlmodelc

# Specify model and camera ID
cargo run --release -- models/movenet_multipose.mlpackage 1
```

### Controls

- **q**: Quit the application

## How It Works

1. **Hardware Acceleration**: CoreML leverages Apple's Neural Engine and GPU via the `coreml-rs` Rust bindings
2. **Model Loading**: Loads MoveNet Multipose CoreML model (`.mlpackage` or `.mlmodelc` format)
3. **Webcam Capture**: OpenCV-Rust captures frames using AVFoundation backend
4. **Preprocessing**: Frames resized to 256x256 and converted to FP16 format
5. **Inference**: Model predicts 17 body keypoints for up to 6 people simultaneously
6. **Temporal Smoothing**: Exponential moving average reduces jitter in keypoint positions
7. **Visualization**: Draws full skeleton with color-coded confidence levels
8. **Display**: Shows annotated frame with minimal overhead for maximum FPS

## Keypoint Detection

MoveNet detects 17 body keypoints per person (COCO format):

- **Face**: Nose, Eyes (L/R), Ears (L/R)
- **Upper Body**: Shoulders (L/R), Elbows (L/R), Wrists (L/R)
- **Lower Body**: Hips (L/R), Knees (L/R), **Ankles** (L/R)

Each keypoint contains:

- `y` coordinate (normalized 0-1)
- `x` coordinate (normalized 0-1)
- `confidence` score (0-1)

The visualization uses color-coded keypoints:

- **Red** (low confidence) → **Orange** → **Yellow** → **Green** (high confidence)

## Project Structure

```
src/
  main.rs              - Application entry point and main loop
  pose_detector.rs     - CoreML pose detection with temporal smoothing
  visualization.rs     - Drawing keypoints and skeleton connections

coreml-rs/             - Custom Rust bindings for CoreML framework
  src/lib.rs           - Main CoreML interface
  swift-library/       - Swift wrapper for CoreML API

models/                - CoreML model files
  movenet_multipose.mlpackage/    - MoveNet Multipose model

convert_coreml.py      - Script to convert ONNX models to CoreML
Cargo.toml             - Rust dependencies and build configuration
```

## Tech Stack

- **Language**: Rust (2021 edition)
- **Computer Vision**: OpenCV 0.95 (Rust bindings)
- **ML Framework**: CoreML via custom `coreml-rs` bindings
- **Array Processing**: ndarray 0.16
- **Error Handling**: anyhow 1.0
- **FP16 Support**: half 2.3

### Why Rust?

The project was rewritten from C++ to Rust for several benefits:

- **Memory Safety**: Eliminates common C++ issues like null pointer dereferences and buffer overflows
- **Performance**: Zero-cost abstractions and excellent optimization
- **Ergonomics**: Modern language features (Result types, pattern matching, traits)
- **Ecosystem**: Cargo for dependency management and building
- **Concurrency**: Built-in safety for future multi-threading enhancements

## Development

### API Documentation

See [API.md](API.md) for detailed documentation on using the pose detection API in your own Rust projects.

### Running Tests

```bash
cargo test
```

### Checking Code

```bash
# Run clippy for lints
cargo clippy

# Format code
cargo fmt

# Check without building
cargo check
```

## Troubleshooting

**Rust compilation errors:**

- Ensure Rust is up to date: `rustup update`
- Check OpenCV installation: `brew info opencv`
- Set OpenCV path if needed: `export OPENCV_LINK_PATHS=/opt/homebrew/lib`

**Webcam not opening:**

- Check camera permissions: System Settings > Privacy & Security > Camera
- Allow Terminal (or your IDE) to access the camera
- Try different camera: `cargo run --release -- models/movenet_multipose.mlpackage 1`
- Use QuickTime Player to verify camera works

**Model file not found:**

- Ensure model is in `models/` directory
- Use either `.mlpackage` or `.mlmodelc` format
- Or specify full path: `cargo run --release -- /path/to/model.mlpackage`

**Low FPS / Performance issues:**

- Use release mode: `cargo run --release` (not just `cargo run`)
- Ensure CoreML is working (check console for initialization messages)
- Close other applications using camera/GPU
- Check Activity Monitor for high CPU/memory usage

**CoreML initialization fails:**

- Update to latest macOS version (CoreML support varies by OS version)
- Verify model format is correct (`.mlpackage` or `.mlmodelc`)
- Check model was converted properly for your macOS version

## Additional Resources

- **API Documentation**: See [API.md](API.md) for detailed API usage and code examples
- **MoveNet Model**: [Google MoveNet on Kaggle](https://www.kaggle.com/models/google/movenet)
- **CoreML Documentation**: [Apple CoreML](https://developer.apple.com/documentation/coreml)
- **OpenCV Rust**: [opencv-rust crate](https://crates.io/crates/opencv)

## License

This project uses the MoveNet model which is provided by Google under the Apache 2.0 license.
