mod pose_detector;
mod visualization;
mod footstep_tracker;
mod person_detector;

use anyhow::{ Context, Result };
use opencv::{ highgui, prelude::*, videoio };
use pose_detector::{ PoseDetector, PoseDetectorConfig, Keypoints };
use visualization::{ draw_all_keypoints, draw_all_ankles, draw_footsteps, draw_bounding_boxes };
use footstep_tracker::FootstepTracker;
use person_detector::{ YoloDetector, YoloDetectorConfig, BoundingBox, PersonTracker };

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
        "models/rtmpose.mlpackage".to_string()
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

    // Configure and initialize the YOLO person detector
    let yolo_config = YoloDetectorConfig::default();
    let mut person_detector = YoloDetector::new(yolo_config).context(
        "Failed to initialize YOLO detector"
    )?;

    // Configure and initialize the pose detector
    let config = PoseDetectorConfig {
        model_path,
        ..Default::default()
    };

    let mut pose_detector = PoseDetector::new(config).context(
        "Failed to initialize pose detector"
    )?;

    // Initialize footstep tracker (footsteps visible for 5 seconds)
    let mut footstep_tracker = FootstepTracker::new(5);

    // Stable ID tracker for person boxes
    let mut id_tracker = PersonTracker::new(30);

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

        // Stage 1: Detect people with YOLO
        #[cfg(feature = "debug")]
        let start = std::time::Instant::now();

        let people = person_detector.detect_people(&frame)?;
        let people_with_ids = id_tracker.assign_ids(people);

        #[cfg(feature = "debug")]
        {
            let yolo_time = start.elapsed();
            if pose_detector.frame_counter % 30 == 0 {
                println!("YOLO detection time: {:.2?}", yolo_time);
            }
        }

        // Stage 2: Run pose detection on each person's bounding box
        let mut all_keypoints: Vec<Keypoints> = Vec::new();
        let mut expanded_bboxes: Vec<BoundingBox> = Vec::new();

        let mut keyed_keypoints: Vec<(usize, Keypoints)> = Vec::new();

        for (person_idx, (person_id, bbox)) in people_with_ids.iter().enumerate() {
            // Expand bounding box slightly for better coverage
            let expanded_bbox = bbox.expand(1.4);
            expanded_bboxes.push(expanded_bbox.clone());

            // Crop the person region
            let person_crop = YoloDetector::crop_region(
                &frame,
                &expanded_bbox,
                pose_detector.config.input_height
            )?;

            // Detect pose in the cropped region
            let mut person_keypoints_batch = pose_detector.detect_pose(&person_crop)?;

            // Transform keypoints back to original frame coordinates
            if let Some(person_keypoints) = person_keypoints_batch.first_mut() {
                let (x, y, w, h) = expanded_bbox.to_pixels(frame.cols(), frame.rows());

                for kp in person_keypoints.iter_mut() {
                    // kp[0] is y (normalized), kp[1] is x (normalized)
                    // Convert from crop coordinates to frame coordinates
                    kp[1] = (kp[1] * (w as f32) + (x as f32)) / (frame.cols() as f32);
                    kp[0] = (kp[0] * (h as f32) + (y as f32)) / (frame.rows() as f32);
                }

                keyed_keypoints.push((*person_id, person_keypoints.clone()));
            }

            #[cfg(feature = "debug")]
            {
                if pose_detector.frame_counter % 30 == 0 {
                    println!("Person {}: bbox confidence={:.2}", person_idx, bbox.confidence);
                }
            }
        }

        #[cfg(feature = "debug")]
        {
            if pose_detector.frame_counter % 30 == 0 {
                let total_time = start.elapsed();
                println!(
                    "Total detection time (YOLO + {} poses): {:.2?}",
                    people.len(),
                    total_time
                );
            }
        }

        // Update footstep tracking
        footstep_tracker.update(&keyed_keypoints);

        // Draw expanded bounding boxes (used for pose detection)
        draw_bounding_boxes(&mut frame, &people_with_ids)?;

        // Visualize all keypoints
        // draw_all_keypoints(&mut frame, &all_keypoints, 0.1)?;
        // draw_all_ankles(&mut frame, &all_keypoints, 0.1)?;

        // Draw footsteps
        let all_footsteps = footstep_tracker.get_all_footsteps();
        draw_footsteps(&mut frame, &all_footsteps)?;

        highgui::imshow(window_name, &frame)?;

        // Break on 'q' key - use appropriate delay for video playback
        if highgui::wait_key(1)? == (b'q' as i32) {
            break;
        }
    }

    Ok(())
}
