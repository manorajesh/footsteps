use crate::pose_detector::{ Keypoint, MultiPoseKeypoints };
use std::collections::HashMap;

const COOLDOWN_FRAMES: usize = 10;
const MIN_CONFIDENCE: f32 = 0.2;

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

/// Tracks footsteps for a single person
#[derive(Debug)]
struct PersonFootstepTracker {
    /// Previous ankle positions [left_y, left_x, right_y, right_x]
    prev_left_ankle: Option<(f32, f32, f32)>, // (y, x, confidence)
    prev_right_ankle: Option<(f32, f32, f32)>,
    /// Detected footsteps for this person
    footsteps: Vec<Footstep>,
    /// Vertical velocity for each ankle (to detect downward motion stopping)
    left_velocity: f32,
    right_velocity: f32,
    /// Flags to track if we've recently detected a step (prevent duplicates)
    left_cooldown: usize,
    right_cooldown: usize,
}

impl PersonFootstepTracker {
    fn new() -> Self {
        Self {
            prev_left_ankle: None,
            prev_right_ankle: None,
            footsteps: Vec::new(),
            left_velocity: 0.0,
            right_velocity: 0.0,
            left_cooldown: 0,
            right_cooldown: 0,
        }
    }

    /// Update tracking with new ankle positions and detect footsteps
    fn update(&mut self, left_ankle: (f32, f32, f32), right_ankle: (f32, f32, f32)) {
        // Decay cooldowns
        if self.left_cooldown > 0 {
            self.left_cooldown -= 1;
        }
        if self.right_cooldown > 0 {
            self.right_cooldown -= 1;
        }

        // Track left ankle
        if left_ankle.2 > MIN_CONFIDENCE {
            if let Some(prev) = self.prev_left_ankle {
                // Calculate vertical velocity (positive = moving down)
                let new_velocity = left_ankle.0 - prev.0;

                // Detect footstep: was moving down (velocity > 0), now stopped/slowed (velocity ~= 0)
                // and ankle is in lower part of frame (y > 0.5)
                if
                    self.left_velocity > 0.005 &&
                    new_velocity.abs() < 0.003 &&
                    left_ankle.0 > 0.5 &&
                    self.left_cooldown == 0
                {
                    self.footsteps.push(Footstep {
                        x: left_ankle.1,
                        y: left_ankle.0,
                        timestamp: std::time::Instant::now(),
                        foot: Foot::Left,
                    });
                    self.left_cooldown = COOLDOWN_FRAMES; // Prevent duplicate detection for ~10 frames
                }

                self.left_velocity = new_velocity;
            }
            self.prev_left_ankle = Some(left_ankle);
        }

        // Track right ankle
        if right_ankle.2 > MIN_CONFIDENCE {
            if let Some(prev) = self.prev_right_ankle {
                let new_velocity = right_ankle.0 - prev.0;

                if
                    self.right_velocity > 0.005 &&
                    new_velocity.abs() < 0.003 &&
                    right_ankle.0 > 0.5 &&
                    self.right_cooldown == 0
                {
                    self.footsteps.push(Footstep {
                        x: right_ankle.1,
                        y: right_ankle.0,
                        timestamp: std::time::Instant::now(),
                        foot: Foot::Right,
                    });
                    self.right_cooldown = COOLDOWN_FRAMES;
                }

                self.right_velocity = new_velocity;
            }
            self.prev_right_ankle = Some(right_ankle);
        }
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

    /// Update with new frame's keypoints
    pub fn update(&mut self, all_keypoints: &MultiPoseKeypoints) {
        // Match current detections to existing trackers based on person index
        // In a more sophisticated system, we'd match based on position/ID

        for (person_idx, person_keypoints) in all_keypoints.iter().enumerate() {
            // Get or create tracker for this person
            let tracker = self.person_trackers
                .entry(person_idx)
                .or_insert_with(PersonFootstepTracker::new);

            // Extract ankle positions
            let left_ankle = person_keypoints[Keypoint::LeftAnkle as usize];
            let right_ankle = person_keypoints[Keypoint::RightAnkle as usize];

            // Update tracker
            tracker.update(
                (left_ankle[0], left_ankle[1], left_ankle[2]),
                (right_ankle[0], right_ankle[1], right_ankle[2])
            );

            // Cleanup old footsteps
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
