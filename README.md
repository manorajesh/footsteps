# Footstep Tracker

A real-time footstep tracking application using MoveNet Lightning model with ONNX Runtime and OpenCV in C++.

## Features

- Real-time pose estimation using MoveNet Lightning
- Webcam capture and processing
- Visual tracking of all body keypoints
- Special highlighting of ankle positions for footstep tracking
- FPS counter for performance monitoring

## Prerequisites

### macOS

Install dependencies using Homebrew:

```bash
# Install OpenCV
brew install opencv

# Install ONNX Runtime
brew install onnxruntime
```

### Other Platforms

- **OpenCV**: Download from [opencv.org](https://opencv.org/)
- **ONNX Runtime**: Download from [github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases)

## Download the MoveNet Model

1. Create a `models` directory in the project root:

   ```bash
   mkdir -p models
   ```

2. Download the MoveNet Lightning model in ONNX format:

   You can convert the TensorFlow Lite model to ONNX, or download a pre-converted version:

   ```bash
   cd models
   # Download from TensorFlow Hub and convert, or use a pre-converted ONNX model
   # Place the model as: movenet_lightning.onnx
   ```

   **Option 1: Using TensorFlow Hub (requires Python)**

   ```bash
   pip install tensorflow tensorflow-hub onnx tf2onnx
   ```

   Then create a conversion script `convert_model.py`:

   ```python
   import tensorflow as tf
   import tensorflow_hub as hub
   import tf2onnx

   # Load MoveNet Lightning from TensorFlow Hub
   model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
   movenet = model.signatures['serving_default']

   # Convert to ONNX
   spec = (tf.TensorSpec((1, 192, 192, 3), tf.int32, name="input"),)
   output_path = "models/movenet_lightning.onnx"

   model_proto, _ = tf2onnx.convert.from_function(
       movenet,
       input_signature=spec,
       opset=13,
       output_path=output_path
   )

   print(f"Model saved to {output_path}")
   ```

   Run the conversion:

   ```bash
   python convert_model.py
   ```

   **Option 2: Pre-converted model**

   Search for "movenet lightning onnx" to find pre-converted versions, or use other pose estimation ONNX models compatible with the same input format (192x192 RGB).

## Building the Project

1. Create a build directory:

   ```bash
   mkdir build
   cd build
   ```

2. Configure with CMake:

   ```bash
   cmake ..
   ```

   If ONNX Runtime is not found automatically, specify its location:

   ```bash
   cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..
   ```

3. Build the project:
   ```bash
   cmake --build .
   ```

## Running

From the build directory:

```bash
./footstep_tracker
```

Or specify a custom model path:

```bash
./footstep_tracker ../models/movenet_lightning.onnx
```

### Controls

- **q**: Quit the application

## Project Structure

```
.
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── src/
│   └── main.cpp            # Main application code
├── models/
│   └── movenet_lightning.onnx  # MoveNet model (download separately)
└── build/                  # Build directory (created during build)
```

## How It Works

1. **Model Loading**: The application loads the MoveNet Lightning ONNX model using ONNX Runtime
2. **Webcam Capture**: OpenCV captures frames from your default webcam
3. **Preprocessing**: Each frame is resized to 192x192 and normalized
4. **Inference**: The model predicts 17 body keypoints (including ankles)
5. **Visualization**: Keypoints and skeleton are drawn on the frame, with special emphasis on ankle positions
6. **Display**: The annotated frame is displayed in real-time

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

- Make sure it's installed via Homebrew: `brew install onnxruntime`
- Or set the path manually: `cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..`

**Webcam not opening:**

- Check camera permissions in System Preferences > Security & Privacy > Camera
- Try a different camera index: Modify `cv::VideoCapture cap(0);` to `cap(1);` in main.cpp

**Model file not found:**

- Ensure the model is placed at `models/movenet_lightning.onnx`
- Or provide the path as a command line argument

## Next Steps

This starter code provides the foundation for footstep tracking. You can extend it to:

- Track ankle positions over time
- Detect footsteps based on ankle movement and ground contact
- Count steps
- Analyze gait patterns
- Record footstep data for analysis

## License

This is starter code for educational purposes. MoveNet model is from Google and subject to its license terms.
