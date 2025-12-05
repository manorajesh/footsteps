mod pose_detector;
mod visualization;

use anyhow::{ Context, Result };
use opencv::{ highgui, prelude::*, videoio };
use pose_detector::{ PoseDetector, PoseDetectorConfig };
use visualization::draw_all_keypoints;

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let model_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "models/movenet_multipose.mlpackage".to_string()
    };

    let camera_id = if args.len() > 2 { args[2].parse().unwrap_or(0) } else { 0 };

    // Configure and initialize the pose detector
    let config = PoseDetectorConfig {
        model_path,
        ..Default::default()
    };

    let mut detector = PoseDetector::new(config).context("Failed to initialize pose detector")?;

    // Open webcam with AVFoundation backend (better for macOS)
    let mut cap = videoio::VideoCapture
        ::new(camera_id, videoio::CAP_AVFOUNDATION)
        .context("Failed to open video capture")?;

    if !videoio::VideoCapture::is_opened(&cap)? {
        eprintln!("Error: Could not open camera {}", camera_id);
        eprintln!(
            "Try a different camera ID with: cargo run --release -- models/movenet_multipose.mlpackage <camera_id>"
        );
        return Ok(());
    }

    println!("Press 'q' to quit");

    let window_name = "Footstep Tracker - CoreML";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    loop {
        let mut frame = Mat::default();
        cap.read(&mut frame)?;

        if frame.empty() {
            break;
        }

        // Detect pose and get keypoints for all people
        let all_keypoints = detector.detect_pose(&frame)?;

        // Visualize all keypoints
        draw_all_keypoints(&mut frame, &all_keypoints, 0.1)?;

        highgui::imshow(window_name, &frame)?;

        // Break on 'q' key
        if highgui::wait_key(1)? == (b'q' as i32) {
            break;
        }
    }

    Ok(())
}
