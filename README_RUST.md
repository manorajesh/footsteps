# Footstep Tracker (Rust)

A real-time footstep tracking application using MoveNet pose estimation with ONNX Runtime (CoreML accelerated) and OpenCV, rewritten in Rust.

## Features

- Real-time pose estimation using MoveNet Thunder/Lightning models
- Hardware acceleration via Apple's CoreML (Neural Engine + GPU)
- Webcam capture and processing with AVFoundation backend
- Visual tracking of leg keypoints and ankle positions
- Optimized for performance on Apple Silicon Macs
- Written in safe, idiomatic Rust

## Prerequisites

### macOS (Apple Silicon)

1. **Rust**: Install from [rustup.rs](https://rustup.rs/)

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **OpenCV**: Install via Homebrew

   ```bash
   brew install opencv
   ```

3. **ONNX Runtime with CoreML**:

   - The existing `dependencies/` folder with ONNX Runtime from the C++ setup will work
   - Or download from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
   - Version: 1.23.2 or later
   - File: `onnxruntime-osx-arm64-{version}.tgz`
   - Extract in `dependencies/` folder

4. **LLVM**: Required for OpenCV bindings (provides libclang)

   ```bash
   brew install llvm
   ```

   The project's `.cargo/config.toml` automatically sets `LIBCLANG_PATH` to `/opt/homebrew/opt/llvm/lib`

## Setup

If you haven't already run the C++ setup, download the MoveNet models:

```bash
# The models should already be in the models/ directory
# If not, download from Kaggle:
# https://www.kaggle.com/models/google/movenet
```

## Building

### Development Build (with debug logging)

```bash
cargo build --features debug
```

### Release Build (optimized)

```bash
cargo build --release
```

The release build is highly optimized with:

- Optimization level 3 (`opt-level = 3`)
- Link-time optimization (`lto = true`)
- Single codegen unit for maximum optimization

## Running

### Basic Usage

```bash
# Development mode
cargo run --release

# Or run the binary directly
./target/release/footstep-tracker
```

### With Custom Model

```bash
cargo run --release -- models/rtmo-s.onnx
```

### With Custom Camera

```bash
cargo run --release -- models/movenet_multipose.onnx 1
```

### With Debug Logging

```bash
cargo run --release --features debug
```

This will show:

- Initialization messages
- CoreML status
- Model info
- Inference timing (every 30 frames)

## Environment Variables

### ONNX Runtime Path (Optional)

If ONNX Runtime is not in `dependencies/`, set:

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
cargo build --release
```

### OpenCV Path (Optional)

The `opencv` crate usually auto-detects Homebrew installations, but if needed:

```bash
export OpenCV_DIR=/opt/homebrew/opt/opencv
cargo build --release
```

## Project Structure

```
src/
├── main.rs              # Main application entry point
├── pose_detector.rs     # Pose detection logic with ONNX Runtime
└── visualization.rs     # Drawing keypoints and skeletons
Cargo.toml              # Dependencies and build configuration
build.rs                # Build script for linking ONNX Runtime
.cargo/
└── config.toml         # Cargo configuration for linking
```

## Key Dependencies

- **opencv**: Rust bindings for OpenCV (camera I/O, image processing, display)
- **ort**: ONNX Runtime bindings with CoreML support
- **anyhow**: Error handling

## Performance

On Apple M1/M2/M3:

- MoveNet MultiPose: ~30-40 FPS with CoreML
- Similar or better performance compared to C++ version
- Zero-cost abstractions maintain C++-like performance

## Differences from C++ Version

1. **Memory Safety**: Rust's ownership system prevents memory leaks and data races
2. **Error Handling**: Uses `Result` and `anyhow` for ergonomic error propagation
3. **Pattern Matching**: Enum-based keypoint indexing is type-safe
4. **No Manual Memory Management**: No need for manual resource cleanup
5. **Cargo Build System**: Simpler dependency management than CMake

## Troubleshooting

### OpenCV linking errors

```bash
brew reinstall opencv
export OpenCV_DIR=/opt/homebrew/opt/opencv
```

### ONNX Runtime not found

Ensure the library is in one of these locations:

- `dependencies/onnxruntime-osx-arm64-*/lib/`
- `/opt/homebrew/lib/`
- Custom path via `ONNXRUNTIME_ROOT`

### CoreML not working

CoreML requires macOS 12+ and is only available on Apple Silicon. The application will automatically fall back to CPU if CoreML initialization fails.

### Camera access denied

Grant camera permissions to Terminal or your terminal app in:
**System Settings → Privacy & Security → Camera**

## License

Same as the original C++ version.
