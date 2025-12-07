mod pose_detector;
mod visualization;
mod footstep_tracker;

use anyhow::{ Context, Result };
use opencv::{ highgui, prelude::*, videoio };
use pose_detector::{ PoseDetector, PoseDetectorConfig };
use visualization::{ draw_all_keypoints, draw_footsteps };
use footstep_tracker::FootstepTracker;

enum VideoSource {
    Webcam(i32),
    File(String),
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let model_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "models/movenet_singlepose_lightning.mlpackage".to_string()
    };

    // Determine video source: file path or camera ID
    let video_source = if args.len() > 2 {
        let arg = &args[2];
        // Check if it's a file path (contains '.' or '/')
        if arg.contains('.') || arg.contains('/') {
            VideoSource::File(arg.clone())
        } else {
            // Try to parse as camera ID
            let camera_id = arg.parse().unwrap_or(0);
            VideoSource::Webcam(camera_id)
        }
    } else {
        VideoSource::Webcam(0)
    };

    // Configure and initialize the pose detector
    let config = PoseDetectorConfig {
        model_path,
        ..Default::default()
    };

    let mut detector = PoseDetector::new(config).context("Failed to initialize pose detector")?;

    // Initialize footstep tracker (footsteps visible for 5 seconds)
    let mut footstep_tracker = FootstepTracker::new(5);

    // Open video source and get FPS for video files
    let (mut cap, video_fps) = match &video_source {
        VideoSource::Webcam(camera_id) => {
            println!("Opening camera {}...", camera_id);
            let cap = videoio::VideoCapture
                ::new(*camera_id, videoio::CAP_AVFOUNDATION)
                .context("Failed to open video capture")?;

            if !videoio::VideoCapture::is_opened(&cap)? {
                eprintln!("Error: Could not open camera {}", camera_id);
                eprintln!(
                    "Try a different camera ID with: cargo run --release -- <model_path> <camera_id>"
                );
                return Ok(());
            }
            (cap, None)
        }
        VideoSource::File(path) => {
            println!("Opening video file: {}", path);
            let cap = videoio::VideoCapture
                ::from_file(path, videoio::CAP_ANY)
                .context("Failed to open video file")?;

            if !videoio::VideoCapture::is_opened(&cap)? {
                eprintln!("Error: Could not open video file: {}", path);
                eprintln!("Make sure the file exists and is a valid video format");
                return Ok(());
            }

            let fps = cap.get(videoio::CAP_PROP_FPS)?;
            let frame_count = cap.get(videoio::CAP_PROP_FRAME_COUNT)?;
            println!("Video: {:.1} FPS, {} frames", fps, frame_count as i32);
            (cap, Some(fps))
        }
    };

    let is_video_file = matches!(video_source, VideoSource::File(_));

    // Calculate delay for video playback (in milliseconds)
    let frame_delay = if let Some(fps) = video_fps {
        if fps > 0.0 {
            (1000.0 / fps) as i32
        } else {
            30 // fallback to ~30 FPS
        }
    } else {
        1 // For webcam, minimal delay
    };

    println!("Press 'q' to quit");

    let window_name = "Footstep Tracker - CoreML";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    loop {
        let mut frame = Mat::default();
        cap.read(&mut frame)?;

        if frame.empty() {
            // If it's a video file, loop back to the beginning
            if is_video_file {
                println!("Looping video...");
                cap.set(videoio::CAP_PROP_POS_FRAMES, 0.0)?;
                cap.read(&mut frame)?;

                // If still empty after reset, break
                if frame.empty() {
                    break;
                }
            } else {
                break;
            }
        }

        // Detect pose and get keypoints for all people
        #[cfg(feature = "debug")]
        let start = std::time::Instant::now();

        let all_keypoints = detector.detect_pose(&frame)?;

        #[cfg(feature = "debug")]
        {
            if detector.frame_counter % 30 == 0 {
                let duration = start.elapsed();
                println!("End-to-End Detection time: {:.2?}", duration);
            }
        }

        // Update footstep tracking
        footstep_tracker.update(&all_keypoints);

        // Visualize all keypoints
        draw_all_keypoints(&mut frame, &all_keypoints, 0.1)?;

        // Draw footsteps
        let all_footsteps = footstep_tracker.get_all_footsteps();
        draw_footsteps(&mut frame, &all_footsteps)?;

        highgui::imshow(window_name, &frame)?;

        // Break on 'q' key - use appropriate delay for video playback
        if highgui::wait_key(frame_delay)? == (b'q' as i32) {
            break;
        }
    }

    Ok(())
}
