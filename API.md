# Pose Detector API

This project is written in Rust and provides a clean, extensible module structure. The core pose detection functionality is encapsulated in the `PoseDetector` struct with configurable options and temporal smoothing for stable tracking.

## Architecture

- **`src/pose_detector.rs`** - Core pose detection with CoreML integration
- **`src/visualization.rs`** - Helper functions for drawing keypoints and skeletons
- **`src/main.rs`** - Example application with webcam capture

## Basic Usage

```rust
use pose_detector::{PoseDetector, PoseDetectorConfig};
use opencv::prelude::*;

// Create with default configuration
let config = PoseDetectorConfig::default();
let mut detector = PoseDetector::new(config)?;

// Process a frame
let frame: Mat = /* ... */;
let all_keypoints = detector.detect_pose(&frame)?;

// Access keypoint data for first person
if let Some(keypoints) = all_keypoints.first() {
    // keypoints[i] = [y, x, confidence]
    let left_ankle_y = keypoints[Keypoint::LeftAnkle as usize][0];
    let left_ankle_x = keypoints[Keypoint::LeftAnkle as usize][1];
    let confidence = keypoints[Keypoint::LeftAnkle as usize][2];
}
```

## Configuration Options

```rust
use pose_detector::PoseDetectorConfig;

let config = PoseDetectorConfig {
    model_path: "models/movenet_multipose.mlpackage".to_string(),
    model_input_size: 256,                    // Input resolution (256x256)
    num_keypoints: 17,                        // COCO format keypoints
    max_people: 6,                            // Maximum people to detect
    low_confidence_threshold: 0.5,            // Threshold for extra smoothing
    smoothing_alpha: 0.3,                     // Normal smoothing factor
    low_confidence_alpha: 0.7,                // Aggressive smoothing for low confidence
};

let mut detector = PoseDetector::new(config)?;
```

### Temporal Smoothing

The detector includes built-in temporal smoothing using exponential moving average:

- **`smoothing_alpha`**: Controls normal smoothing (0.0 = no smoothing, 1.0 = maximum smoothing)
- **`low_confidence_alpha`**: Used when confidence drops below threshold for more aggressive smoothing
- This reduces jitter and provides stable keypoint tracking across frames

## Multi-Person Detection

The detector returns keypoints for all detected people:

```rust
let all_keypoints = detector.detect_pose(&frame)?;

for (person_idx, keypoints) in all_keypoints.iter().enumerate() {
    println!("Person {}: {} keypoints detected", person_idx, keypoints.len());

    // Process each person's keypoints
    for (kp_idx, keypoint) in keypoints.iter().enumerate() {
        let [y, x, confidence] = keypoint;
        println!("  Keypoint {}: ({:.2}, {:.2}) confidence: {:.2}",
                 kp_idx, x, y, confidence);
    }
}
```

## Keypoint Format

Each keypoint is a `[f32; 3]` array containing:

- `[0]` - Y coordinate (normalized 0.0-1.0)
- `[1]` - X coordinate (normalized 0.0-1.0)
- `[2]` - Confidence score (0.0-1.0)

### Type Definitions

```rust
pub type KeypointData = [f32; 3];
pub type Keypoints = Vec<KeypointData>;           // One person
pub type MultiPoseKeypoints = Vec<Keypoints>;     // Multiple people
```

### Available Keypoints (17 total)

```rust
pub enum Keypoint {
    Nose = 0,
    LeftEye = 1, RightEye = 2,
    LeftEar = 3, RightEar = 4,
    LeftShoulder = 5, RightShoulder = 6,
    LeftElbow = 7, RightElbow = 8,
    LeftWrist = 9, RightWrist = 10,
    LeftHip = 11, RightHip = 12,
    LeftKnee = 13, RightKnee = 14,
    LeftAnkle = 15, RightAnkle = 16,
}
```

## Visualization Functions

The `visualization` module provides drawing functions:

```rust
use visualization::{draw_all_keypoints, draw_all_ankles};

// Draw full skeleton with all keypoints for all people
draw_all_keypoints(&mut frame, &all_keypoints, 0.1)?;

// Draw only ankles (optimized for footstep tracking)
draw_all_ankles(&mut frame, &all_keypoints, 0.1)?;
```

### Color Coding

Keypoints are color-coded by confidence:

- **Red** (confidence < 0.5): Low confidence
- **Orange-Yellow** (0.5 ≤ confidence < 0.75): Medium confidence
- **Green** (confidence ≥ 0.75): High confidence

## Example: Custom Processing

```rust
use anyhow::Result;
use opencv::{highgui, prelude::*, videoio};
use pose_detector::{Keypoint, PoseDetector, PoseDetectorConfig};

fn main() -> Result<()> {
    // Setup detector
    let config = PoseDetectorConfig {
        model_path: "models/movenet_multipose.mlpackage".to_string(),
        ..Default::default()
    };
    let mut detector = PoseDetector::new(config)?;

    // Open webcam
    let mut cap = videoio::VideoCapture::new(0, videoio::CAP_AVFOUNDATION)?;
    let mut frame = Mat::default();

    while cap.read(&mut frame)? {
        // Get keypoints for all people
        let all_keypoints = detector.detect_pose(&frame)?;

        // Custom processing - e.g., track ankle positions
        for (person_idx, keypoints) in all_keypoints.iter().enumerate() {
            let left_ankle = &keypoints[Keypoint::LeftAnkle as usize];

            if left_ankle[2] > 0.5 {
                let ankle_x = left_ankle[1] * (frame.cols() as f32);
                let ankle_y = left_ankle[0] * (frame.rows() as f32);

                println!("Person {}: Left ankle at: ({:.1}, {:.1})",
                         person_idx, ankle_x, ankle_y);
            }
        }

        // Visualize or process further
        highgui::imshow("Frame", &frame)?;
        if highgui::wait_key(1)? == b'q' as i32 {
            break;
        }
    }

    Ok(())
}
```

## Building and Running

The project uses Cargo (Rust's build system):

```bash
# Development build (fast compilation)
cargo build

# Release build (optimized)
cargo build --release

# Run directly
cargo run --release -- [model_path] [camera_id]
```

## Extending the Project

To add new functionality:

1. **Custom Pose Analysis** - Process `MultiPoseKeypoints` for your own tracking algorithms
2. **Different Visualization** - Create new drawing functions in `src/visualization.rs`
3. **Data Export** - Serialize keypoints to JSON/CSV for analysis
4. **Multi-frame Analysis** - Store keypoints over time for gesture/gait recognition
5. **Integration** - Use `PoseDetector` as a module in larger Rust applications

### Example: Footstep Detection

```rust
fn detect_footstep(keypoints: &Keypoints, frame_height: i32) -> bool {
    let left_ankle = &keypoints[Keypoint::LeftAnkle as usize];
    let right_ankle = &keypoints[Keypoint::RightAnkle as usize];

    // Check if ankles are detected with good confidence
    if left_ankle[2] > 0.6 && right_ankle[2] > 0.6 {
        let left_y = left_ankle[0] * (frame_height as f32);
        let right_y = right_ankle[0] * (frame_height as f32);

        // Detect if one foot is significantly higher (potential step)
        let height_diff = (left_y - right_y).abs();
        return height_diff > 50.0; // Threshold in pixels
    }

    false
}
```

### Example: Gait Analysis

```rust
struct GaitTracker {
    ankle_history: Vec<Vec<[f32; 2]>>, // Store ankle positions over time
    max_history: usize,
}

impl GaitTracker {
    fn update(&mut self, keypoints: &Keypoints) {
        let left_ankle = &keypoints[Keypoint::LeftAnkle as usize];
        let right_ankle = &keypoints[Keypoint::RightAnkle as usize];

        self.ankle_history.push(vec![
            [left_ankle[0], left_ankle[1]],
            [right_ankle[0], right_ankle[1]],
        ]);

        if self.ankle_history.len() > self.max_history {
            self.ankle_history.remove(0);
        }
    }

    fn calculate_stride_length(&self) -> f32 {
        // Analyze ankle_history to compute stride metrics
        // ... implementation ...
        0.0
    }
}
```

## Dependencies

The project uses the following key crates:

- **`opencv`** (0.95) - OpenCV bindings for video capture and image processing
- **`coreml-rs`** - Custom Rust bindings for Apple's CoreML framework
- **`ndarray`** (0.16) - N-dimensional array handling for model input/output
- **`anyhow`** (1.0) - Error handling
- **`half`** (2.3) - FP16 floating-point support for CoreML

## Performance Considerations

- Always use `--release` mode for real-time performance
- CoreML automatically uses Neural Engine + GPU on Apple Silicon
- Temporal smoothing adds minimal overhead (~1-2% CPU usage)
- Frame processing is typically 30-60 FPS on M1/M2 Macs
- Multi-person detection scales linearly with number of people detected
