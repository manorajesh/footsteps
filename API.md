# Pose Detector API

This project has been refactored into a clean, extensible class structure. The core pose detection functionality is now encapsulated in the `PoseDetector` class with configurable options.

## Architecture

- **`PoseDetector.h`** - Main class header with configuration options
- **`PoseDetector.cpp`** - Implementation of pose detection logic
- **`visualization.h`** - Helper functions for drawing keypoints
- **`main.cpp`** - Simple example application

## Basic Usage

```cpp
#include "PoseDetector.h"

// Create with default configuration
PoseDetector detector;

// Process a frame
cv::Mat frame = ...;
Keypoints keypoints = detector.detectPose(frame);

// Access keypoint data: keypoints[i] = {y, x, confidence}
float left_ankle_y = keypoints[LEFT_ANKLE][0];
float left_ankle_x = keypoints[LEFT_ANKLE][1];
float confidence = keypoints[LEFT_ANKLE][2];
```

## Configuration Options

```cpp
PoseDetectorConfig config;
config.model_path = "models/movenet_lightning.onnx";  // Model to use
config.model_input_size = 192;                        // Input resolution
config.use_coreml = true;                             // Hardware acceleration
config.intra_op_threads = 4;                          // Performance tuning
config.verbose = true;                                // Logging
config.inference_log_frequency = 30;                  // Log every N frames

PoseDetector detector(config);
```

## Accessing Keypoints

The detector exposes keypoints in two ways:

1. **Returned from `detectPose()`** - Each call returns the detected keypoints
2. **Via `getKeypoints()`** - Access the last detected keypoints without re-running inference

```cpp
// Get keypoints from detection
Keypoints kp1 = detector.detectPose(frame);

// Or access the cached result
const Keypoints& kp2 = detector.getKeypoints();
```

## Keypoint Format

Each keypoint is a `std::array<float, 3>` containing:

- `[0]` - Y coordinate (normalized 0-1)
- `[1]` - X coordinate (normalized 0-1)
- `[2]` - Confidence score (0-1)

Available keypoints (17 total):

```
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE
```

## Visualization

Two helper functions are provided in `visualization.h`:

```cpp
#include "visualization.h"

// Draw leg skeleton only (optimized for performance)
drawKeypoints(frame, keypoints, confidence_threshold);

// Draw full body skeleton with all keypoints
drawAllKeypoints(frame, keypoints, confidence_threshold);
```

## Example: Custom Processing

```cpp
#include "PoseDetector.h"
#include <opencv2/opencv.hpp>

int main() {
    // Setup
    PoseDetectorConfig config;
    config.model_path = "models/movenet_thunder.onnx";
    PoseDetector detector(config);

    cv::VideoCapture cap(0);
    cv::Mat frame;

    while (cap.read(frame)) {
        // Get keypoints
        Keypoints kp = detector.detectPose(frame);

        // Custom processing - e.g., track ankle positions
        if (kp[LEFT_ANKLE][2] > 0.5) {
            float ankle_x = kp[LEFT_ANKLE][1] * frame.cols;
            float ankle_y = kp[LEFT_ANKLE][0] * frame.rows;

            // Do something with ankle position
            std::cout << "Left ankle at: " << ankle_x << ", " << ankle_y << std::endl;
        }

        // Visualize or process further
        cv::imshow("Frame", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    return 0;
}
```

## Building

The project uses CMake. After making changes:

```bash
cd build
cmake ..
make
./footstep_tracker [model_path] [camera_id]
```

## Extending the Project

To add new functionality:

1. **Custom Pose Analysis** - Use `detector.getKeypoints()` to access pose data for your own processing
2. **Different Visualization** - Create new drawing functions in `visualization.h`
3. **Data Export** - Access keypoints and export to files/network
4. **Multi-frame Analysis** - Store keypoints over time for gesture/movement recognition
5. **Integration** - Use `PoseDetector` as a component in a larger application
