use crate::pose_detector::{ Keypoint, Keypoints };
use std::collections::HashMap;

const COOLDOWN_FRAMES: usize = 10;
const MIN_CONFIDENCE: f32 = 0.2;
const MOVING_SPEED_THRESH: f32 = 0.01; // ankle clearly moving
const STILL_SPEED_THRESH: f32 = 0.008; // ankle effectively still

/// Represents a single footstep with location and timestamp
#[derive(Debug, Clone)]
pub struct Footstep {
    pub x: f32,
    pub y: f32,
    pub timestamp: std::time::Instant,
    pub foot: Foot,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Foot {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy)]
struct FootMotionState {
    prev_pos: Option<(f32, f32)>,
    prev_speed: f32,
    cooldown: usize,
}

/// Tracks footsteps for a single person
#[derive(Debug)]
struct PersonFootstepTracker {
    footsteps: Vec<Footstep>,
    left: FootMotionState,
    right: FootMotionState,
}

impl PersonFootstepTracker {
    fn new() -> Self {
        Self {
            footsteps: Vec::new(),
            left: FootMotionState { prev_pos: None, prev_speed: 0.0, cooldown: 0 },
            right: FootMotionState { prev_pos: None, prev_speed: 0.0, cooldown: 0 },
        }
    }

    fn update(
        &mut self,
        left_ankle: (f32, f32, f32),
        right_ankle: (f32, f32, f32),
        pelvis: (f32, f32, f32)
    ) {
        if let Some(step) = Self::process_foot(Foot::Left, &mut self.left, left_ankle, pelvis) {
            self.footsteps.push(step);
        }

        if let Some(step) = Self::process_foot(Foot::Right, &mut self.right, right_ankle, pelvis) {
            self.footsteps.push(step);
        }
    }

    fn process_foot(
        foot: Foot,
        motion: &mut FootMotionState,
        ankle: (f32, f32, f32),
        pelvis: (f32, f32, f32)
    ) -> Option<Footstep> {
        if motion.cooldown > 0 {
            motion.cooldown -= 1;
        }

        // Pause state changes if confidence drops
        if ankle.2 < MIN_CONFIDENCE || pelvis.2 < MIN_CONFIDENCE {
            motion.prev_pos = None;
            motion.prev_speed = 0.0;
            return None;
        }
        let pos_abs = (ankle.0, ankle.1);
        let speed = if let Some(prev) = motion.prev_pos {
            let dy = pos_abs.0 - prev.0;
            let dx = pos_abs.1 - prev.1;
            (dy * dy + dx * dx).sqrt()
        } else {
            0.0
        };

        let was_moving = motion.prev_speed > MOVING_SPEED_THRESH;
        let now_still = speed < STILL_SPEED_THRESH;

        let mut maybe_step = None;
        if was_moving && now_still && motion.cooldown == 0 {
            maybe_step = Some(Footstep {
                x: pos_abs.1,
                y: pos_abs.0,
                timestamp: std::time::Instant::now(),
                foot,
            });
            motion.cooldown = COOLDOWN_FRAMES;
        }

        motion.prev_pos = Some(pos_abs);
        motion.prev_speed = speed;

        maybe_step
    }

    /// Get all footsteps for this person
    fn get_footsteps(&self) -> &[Footstep] {
        &self.footsteps
    }

    /// Clean up old footsteps (older than duration)
    fn cleanup_old_footsteps(&mut self, max_age: std::time::Duration) {
        let now = std::time::Instant::now();
        self.footsteps.retain(|step| now.duration_since(step.timestamp) < max_age);
    }
}

/// Tracks footsteps for multiple people
pub struct FootstepTracker {
    /// Map from person index to their tracker
    person_trackers: HashMap<usize, PersonFootstepTracker>,
    /// How long to keep footsteps visible (in seconds)
    footstep_display_duration: std::time::Duration,
}

impl FootstepTracker {
    pub fn new(footstep_display_duration_secs: u64) -> Self {
        Self {
            person_trackers: HashMap::new(),
            footstep_display_duration: std::time::Duration::from_secs(
                footstep_display_duration_secs
            ),
        }
    }

    /// Update with new frame's keypoints paired with stable IDs
    pub fn update(&mut self, keyed_keypoints: &[(usize, Keypoints)]) {
        for (person_id, person_keypoints) in keyed_keypoints.iter() {
            let tracker = self.person_trackers
                .entry(*person_id)
                .or_insert_with(PersonFootstepTracker::new);

            // Extract ankle and hip positions
            let left_ankle = person_keypoints[Keypoint::LeftAnkle as usize];
            let right_ankle = person_keypoints[Keypoint::RightAnkle as usize];
            let left_hip = person_keypoints[Keypoint::LeftHip as usize];
            let right_hip = person_keypoints[Keypoint::RightHip as usize];

            let pelvis = (
                (left_hip[0] + right_hip[0]) * 0.5,
                (left_hip[1] + right_hip[1]) * 0.5,
                (left_hip[2] + right_hip[2]) * 0.5,
            );

            tracker.update(
                (left_ankle[0], left_ankle[1], left_ankle[2]),
                (right_ankle[0], right_ankle[1], right_ankle[2]),
                pelvis
            );

            tracker.cleanup_old_footsteps(self.footstep_display_duration);
        }
    }

    /// Get all footsteps grouped by person
    pub fn get_all_footsteps(&self) -> HashMap<usize, Vec<Footstep>> {
        self.person_trackers
            .iter()
            .map(|(person_idx, tracker)| (*person_idx, tracker.get_footsteps().to_vec()))
            .collect()
    }

    /// Reset all tracking data
    pub fn reset(&mut self) {
        self.person_trackers.clear();
    }
}
