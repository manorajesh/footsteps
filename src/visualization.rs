use crate::pose_detector::{ Keypoint, Keypoints, MultiPoseKeypoints };
use crate::footstep_tracker::{ Footstep, Foot };
use anyhow::Result;
use opencv::{ core, imgproc, prelude::* };
use std::collections::HashMap;

fn confidence_to_color(confidence: f32) -> core::Scalar {
    let confidence = confidence.clamp(0.0, 1.0);

    if confidence < 0.5 {
        let t = confidence * 2.0; // 0.0 to 1.0
        let green = (255.0 * t) as f64;
        core::Scalar::new(0.0, green, 255.0, 0.0) // BGR
    } else {
        let t = (confidence - 0.5) * 2.0; // 0.0 to 1.0
        let red = (255.0 * (1.0 - t)) as f64;
        core::Scalar::new(0.0, 255.0, red, 0.0) // BGR
    }
}

/// Draw ankle keypoints for a single person
pub fn draw_ankle(frame: &mut Mat, keypoints: &Keypoints, confidence_threshold: f32) -> Result<()> {
    let height = frame.rows();
    let width = frame.cols();

    // Draw left ankle
    if keypoints[Keypoint::LeftAnkle as usize][2] > confidence_threshold {
        let x = (keypoints[Keypoint::LeftAnkle as usize][1] * (width as f32)) as i32;
        let y = (keypoints[Keypoint::LeftAnkle as usize][0] * (height as f32)) as i32;
        let confidence = keypoints[Keypoint::LeftAnkle as usize][2];
        let color = confidence_to_color(confidence);
        imgproc::circle(
            frame,
            core::Point::new(x, y),
            20,
            color,
            -1, // Filled
            imgproc::LINE_8,
            0
        )?;
    }

    // Draw right ankle
    if keypoints[Keypoint::RightAnkle as usize][2] > confidence_threshold {
        let x = (keypoints[Keypoint::RightAnkle as usize][1] * (width as f32)) as i32;
        let y = (keypoints[Keypoint::RightAnkle as usize][0] * (height as f32)) as i32;
        let confidence = keypoints[Keypoint::RightAnkle as usize][2];
        let color = confidence_to_color(confidence);
        imgproc::circle(
            frame,
            core::Point::new(x, y),
            20,
            color,
            -1, // Filled
            imgproc::LINE_8,
            0
        )?;
    }

    Ok(())
}

/// Draw ankles for all detected people
pub fn draw_all_ankles(
    frame: &mut Mat,
    all_keypoints: &MultiPoseKeypoints,
    confidence_threshold: f32
) -> Result<()> {
    for keypoints in all_keypoints {
        draw_ankle(frame, keypoints, confidence_threshold)?;
    }
    Ok(())
}

/// Draw all keypoints and skeleton connections for all detected people
pub fn draw_all_keypoints(
    frame: &mut Mat,
    all_keypoints: &MultiPoseKeypoints,
    confidence_threshold: f32
) -> Result<()> {
    let height = frame.rows();
    let width = frame.cols();

    // Define skeleton connections
    let connections: Vec<(Keypoint, Keypoint)> = vec![
        (Keypoint::Nose, Keypoint::LeftEye),
        (Keypoint::Nose, Keypoint::RightEye),
        (Keypoint::LeftEye, Keypoint::LeftEar),
        (Keypoint::RightEye, Keypoint::RightEar),
        (Keypoint::Nose, Keypoint::LeftShoulder),
        (Keypoint::Nose, Keypoint::RightShoulder),
        (Keypoint::LeftShoulder, Keypoint::RightShoulder),
        (Keypoint::LeftShoulder, Keypoint::LeftElbow),
        (Keypoint::LeftElbow, Keypoint::LeftWrist),
        (Keypoint::RightShoulder, Keypoint::RightElbow),
        (Keypoint::RightElbow, Keypoint::RightWrist),
        (Keypoint::LeftShoulder, Keypoint::LeftHip),
        (Keypoint::RightShoulder, Keypoint::RightHip),
        (Keypoint::LeftHip, Keypoint::RightHip),
        (Keypoint::LeftHip, Keypoint::LeftKnee),
        (Keypoint::LeftKnee, Keypoint::LeftAnkle),
        (Keypoint::RightHip, Keypoint::RightKnee),
        (Keypoint::RightKnee, Keypoint::RightAnkle)
    ];

    for keypoints in all_keypoints {
        // Draw all keypoints with confidence-based colors
        for kp in keypoints {
            if kp[2] > confidence_threshold {
                let x = (kp[1] * (width as f32)) as i32;
                let y = (kp[0] * (height as f32)) as i32;
                let confidence = kp[2];
                let color = confidence_to_color(confidence);
                imgproc::circle(
                    frame,
                    core::Point::new(x, y),
                    5,
                    color,
                    -1, // Filled
                    imgproc::LINE_8,
                    0
                )?;
            }
        }

        // Draw skeleton connections
        for (start, end) in &connections {
            let start_idx = *start as usize;
            let end_idx = *end as usize;

            if
                keypoints[start_idx][2] > confidence_threshold &&
                keypoints[end_idx][2] > confidence_threshold
            {
                let x1 = (keypoints[start_idx][1] * (width as f32)) as i32;
                let y1 = (keypoints[start_idx][0] * (height as f32)) as i32;
                let x2 = (keypoints[end_idx][1] * (width as f32)) as i32;
                let y2 = (keypoints[end_idx][0] * (height as f32)) as i32;

                imgproc::line(
                    frame,
                    core::Point::new(x1, y1),
                    core::Point::new(x2, y2),
                    core::Scalar::new(255.0, 0.0, 0.0, 0.0), // Blue (BGR)
                    2,
                    imgproc::LINE_8,
                    0
                )?;
            }
        }
    }

    Ok(())
}

/// Get a unique color for each person based on their index
fn person_color(person_idx: usize) -> core::Scalar {
    // Define a set of distinct colors (BGR format)
    let colors = vec![
        core::Scalar::new(0.0, 255.0, 0.0, 0.0), // Green
        core::Scalar::new(255.0, 0.0, 0.0, 0.0), // Blue
        core::Scalar::new(0.0, 0.0, 255.0, 0.0), // Red
        core::Scalar::new(255.0, 255.0, 0.0, 0.0), // Cyan
        core::Scalar::new(255.0, 0.0, 255.0, 0.0), // Magenta
        core::Scalar::new(0.0, 255.0, 255.0, 0.0), // Yellow
        core::Scalar::new(128.0, 255.0, 0.0, 0.0), // Teal
        core::Scalar::new(0.0, 128.0, 255.0, 0.0) // Orange
    ];

    colors[person_idx % colors.len()]
}

/// Draw footsteps for all people with different colors per person
pub fn draw_footsteps(
    frame: &mut Mat,
    all_footsteps: &HashMap<usize, Vec<Footstep>>
) -> Result<()> {
    let height = frame.rows();
    let width = frame.cols();

    for (person_idx, footsteps) in all_footsteps {
        let color = person_color(*person_idx);

        for footstep in footsteps {
            // Convert normalized coordinates to pixel coordinates
            let x = (footstep.x * (width as f32)) as i32;
            let y = (footstep.y * (height as f32)) as i32;

            // Calculate opacity based on age (fade out old footsteps)
            let age = std::time::Instant::now().duration_since(footstep.timestamp);
            let age_secs = age.as_secs_f32();
            let max_age = 5.0; // Should match footstep_display_duration
            let opacity = 1.0 - (age_secs / max_age).min(1.0);

            // Adjust color alpha based on opacity
            let mut faded_color = color.clone();
            faded_color[0] *= opacity as f64;
            faded_color[1] *= opacity as f64;
            faded_color[2] *= opacity as f64;

            // Draw filled circle for footstep
            let radius = 12;
            imgproc::circle(
                frame,
                core::Point::new(x, y),
                radius,
                faded_color,
                -1, // Filled
                imgproc::LINE_AA,
                0
            )?;

            // Draw a smaller circle in a different color to indicate left vs right foot
            let foot_indicator_color = match footstep.foot {
                Foot::Left => core::Scalar::new(255.0, 255.0, 255.0, 0.0), // White
                Foot::Right => core::Scalar::new(0.0, 0.0, 0.0, 0.0), // Black
            };

            imgproc::circle(
                frame,
                core::Point::new(x, y),
                4,
                foot_indicator_color,
                -1,
                imgproc::LINE_AA,
                0
            )?;
        }
    }

    Ok(())
}
