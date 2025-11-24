# Footstep Tracker

A real-time footstep tracking application using MoveNet pose estimation with ONNX Runtime (CoreML accelerated) and OpenCV in C++.

## Features

- Real-time pose estimation using MoveNet Thunder/Lightning models
- Hardware acceleration via Apple's CoreML (Neural Engine + GPU)
- Webcam capture and processing with AVFoundation backend
- Visual tracking of leg keypoints and ankle positions
- Optimized for performance on Apple Silicon Macs

## Quick Setup (macOS)

Run the automated setup script:

```bash
chmod +x setup.sh
./setup.sh
```

This will:

- Install OpenCV via Homebrew
- Download ONNX Runtime 1.23.2 with CoreML support to `dependencies/` folder
- Download MoveNet Thunder and Lightning models to `models/` directory
- Set up the complete project structure

## Manual Setup

### Prerequisites

#### macOS (Apple Silicon)

Required dependencies:

- **OpenCV**: `brew install opencv`
- **ONNX Runtime with CoreML**: Download from [releases](https://github.com/microsoft/onnxruntime/releases)
  - Version: 1.23.2 or later
  - File: `onnxruntime-osx-arm64-{version}.tgz`
  - Extract in `dependencies/` folder

#### Intel Macs / Other Platforms

- **OpenCV**: Download from [opencv.org](https://opencv.org/)
- **ONNX Runtime**: Download appropriate version from [github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases)

### Download MoveNet Models

The project uses MoveNet pose estimation models in ONNX format. Download from Kaggle:

1. Visit [Kaggle MoveNet Models](https://www.kaggle.com/models/google/movenet)

2. Download ONNX versions:

   - **movenet_thunder.onnx** - More accurate, slower (~30-40 FPS on M1)
   - **movenet_lightning.onnx** - Faster, less accurate (~60+ FPS on M1)

3. Place in the `models/` directory:
   ```bash
   mkdir -p models
   # Move downloaded files to models/
   ```

**Alternative:** Convert from TensorFlow (advanced users only)

## Building the Project

### Quick Build

```bash
mkdir -p build && cd build
cmake ..
make
```

Fully optimized with `-O3`, _with_ logging (default)

### Build With Logging

```bash
cd build
cmake -DENABLE_LOGGING=OFF ..
make
```

Fully optimized with `-O3` **and** logging enabled

- Initialization messages
- CoreML status
- Model info
- Inference timing (every 30 frames)

### Advanced Configuration

**CMake will automatically detect:**

- Local ONNX Runtime (if extracted in `dependencies/` folder)
- Homebrew OpenCV installation

**Manual ONNX Runtime path (if needed):**

```bash
cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime-osx-arm64-1.23.2 ..
```

**Debug build with symbols (for profiling):**

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_LOGGING=ON ..
make
```

**Faster parallel build:**

```bash
make -j$(sysctl -n hw.ncpu)
```

**Clean build when switching configurations:**

```bash
cd build && rm -rf * && cmake -DENABLE_LOGGING=ON .. && make
```

## Running

From the build directory:

```bash
# Run with default model (movenet_thunder.onnx)
./build/footstep_tracker

# Specify model path
./build/footstep_tracker ../models/movenet_lightning.onnx

# Specify model and camera ID
./build/footstep_tracker ../models/movenet_thunder.onnx 0
```

### Controls

- **q**: Quit the application

## How It Works

1. **Hardware Acceleration**: ONNX Runtime with CoreML uses Apple's Neural Engine and GPU
2. **Model Loading**: Loads MoveNet ONNX model (Thunder or Lightning variant)
3. **Webcam Capture**: OpenCV captures frames using AVFoundation backend
4. **Preprocessing**: Frames resized to 256x256 and converted to RGB
5. **Inference**: Model predicts 17 body keypoints with confidence scores
6. **Visualization**: Draws leg skeleton and highlights ankles in real-time
7. **Display**: Shows annotated frame with minimal overhead for maximum FPS

## Keypoint Detection

MoveNet detects 17 body keypoints:

- Nose, Eyes, Ears
- Shoulders, Elbows, Wrists
- Hips, Knees, **Ankles** (highlighted for footstep tracking)

Each keypoint has:

- `x, y` coordinates (normalized 0-1)
- `confidence` score (0-1)

## Troubleshooting

**CMake can't find ONNX Runtime:**

- Run `./setup.sh` to download it automatically to `dependencies/` folder
- Or manually extract `onnxruntime-osx-arm64-{version}.tgz` in `dependencies/` folder
- Or specify path: `cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..`

**Webcam not opening:**

- Check camera permissions: System Settings > Privacy & Security > Camera
- Try different camera: `./footstep_tracker ../models/movenet_thunder.onnx 1`
- Check available cameras: `ls /dev/video*` or use QuickTime to test

**Model file not found:**

- Download from [Kaggle MoveNet Models](https://www.kaggle.com/models/google/movenet)
- Place ONNX files in `models/` directory
- Or specify path: `./footstep_tracker /path/to/model.onnx`

**Low FPS / Performance issues:**

- Ensure CoreML is enabled (check console output for "✓ CoreML acceleration enabled")
- Use Lightning model for better FPS: `./footstep_tracker ../models/movenet_lightning.onnx`
- Close other applications using camera/GPU
- Check Activity Monitor for high CPU usage

**CoreML not working:**

- Update to latest macOS version
- Verify ONNX Runtime version includes CoreML support (1.20.1+)
- Check console for CoreML initialization messages
