use crate::pose_detector::{ Keypoint, Keypoints, MultiPoseKeypoints };
use anyhow::Result;
use opencv::{ core, imgproc, prelude::* };

/// Draw ankle keypoints for a single person
pub fn draw_ankle(frame: &mut Mat, keypoints: &Keypoints, confidence_threshold: f32) -> Result<()> {
    let height = frame.rows();
    let width = frame.cols();

    // Draw left ankle
    if keypoints[Keypoint::LeftAnkle as usize][2] > confidence_threshold {
        let x = (keypoints[Keypoint::LeftAnkle as usize][1] * (width as f32)) as i32;
        let y = (keypoints[Keypoint::LeftAnkle as usize][0] * (height as f32)) as i32;
        imgproc::circle(
            frame,
            core::Point::new(x, y),
            20,
            core::Scalar::new(0.0, 0.0, 255.0, 0.0), // Red (BGR)
            -1, // Filled
            imgproc::LINE_8,
            0
        )?;
    }

    // Draw right ankle
    if keypoints[Keypoint::RightAnkle as usize][2] > confidence_threshold {
        let x = (keypoints[Keypoint::RightAnkle as usize][1] * (width as f32)) as i32;
        let y = (keypoints[Keypoint::RightAnkle as usize][0] * (height as f32)) as i32;
        imgproc::circle(
            frame,
            core::Point::new(x, y),
            20,
            core::Scalar::new(0.0, 0.0, 255.0, 0.0), // Red (BGR)
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
        // Draw all keypoints
        for kp in keypoints {
            if kp[2] > confidence_threshold {
                let x = (kp[1] * (width as f32)) as i32;
                let y = (kp[0] * (height as f32)) as i32;
                imgproc::circle(
                    frame,
                    core::Point::new(x, y),
                    5,
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0), // Green (BGR)
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
