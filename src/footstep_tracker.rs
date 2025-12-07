use crate::pose_detector::{ Keypoint, Keypoints };
use std::collections::{ HashMap, HashSet };
use std::time::{ Duration, Instant };

/// Represents a single footstep with location and timestamp
#[derive(Debug, Clone)]
pub struct Footstep {
    pub x: f32,
    pub y: f32,
    /// Unit vector from the previous footstep of the same foot, if available
    pub direction: Option<(f32, f32)>,
    pub timestamp: Instant,
    pub foot: Foot,
}

/// Footstep event paired with person ID for downstream consumers (e.g., UDP output)
#[derive(Debug, Clone)]
pub struct FootstepEvent {
    pub person_id: usize,
    pub footstep: Footstep,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Foot {
    Left,
    Right,
}

const COOLDOWN_FRAMES: usize = 10;
const MIN_CONFIDENCE: f32 = 0.2;
const MOVING_SPEED_THRESH: f32 = 0.01; // ankle clearly moving
const STILL_SPEED_THRESH: f32 = 0.008; // ankle effectively still

#[derive(Debug, Default)]
struct FootstepHistory {
    history_map: HashMap<usize, Vec<Footstep>>,
    current_trails: HashMap<usize, Vec<Footstep>>,
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
    left: FootMotionState,
    right: FootMotionState,
    last_seen: Instant,
}

impl PersonFootstepTracker {
    fn new() -> Self {
        Self {
            left: FootMotionState { prev_pos: None, prev_speed: 0.0, cooldown: 0 },
            right: FootMotionState { prev_pos: None, prev_speed: 0.0, cooldown: 0 },
            last_seen: Instant::now(),
        }
    }

    fn update(
        &mut self,
        left_ankle: (f32, f32, f32),
        right_ankle: (f32, f32, f32),
        pelvis: (f32, f32, f32)
    ) -> Vec<Footstep> {
        let mut new_steps = Vec::new();

        if let Some(step) = Self::process_foot(Foot::Left, &mut self.left, left_ankle, pelvis) {
            new_steps.push(step);
        }

        if let Some(step) = Self::process_foot(Foot::Right, &mut self.right, right_ankle, pelvis) {
            new_steps.push(step);
        }

        self.last_seen = Instant::now();
        new_steps
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
                direction: None,
                timestamp: std::time::Instant::now(),
                foot,
            });
            motion.cooldown = COOLDOWN_FRAMES;
        }

        motion.prev_pos = Some(pos_abs);
        motion.prev_speed = speed;

        maybe_step
    }
}

impl FootstepHistory {
    fn new() -> Self {
        Self { history_map: HashMap::new(), current_trails: HashMap::new() }
    }

    fn ensure_history(&mut self, person_id: usize) -> &mut Vec<Footstep> {
        self.history_map.entry(person_id).or_insert_with(Vec::new)
    }

    fn ensure_current_trail(&mut self, person_id: usize) -> &mut Vec<Footstep> {
        self.current_trails.entry(person_id).or_insert_with(Vec::new)
    }

    fn match_or_insert(
        &mut self,
        person_id: usize,
        x: f32,
        y: f32,
        foot: Foot,
        timestamp: Instant,
        max_dist: f32
    ) -> Footstep {
        let mut current_step = Footstep { x, y, direction: None, timestamp, foot };

        if
            let Some(prev_step) = self.history_map.get(&person_id).and_then(|steps|
                steps
                    .iter()
                    .rev()
                    .find(|s| s.foot == foot)
            )
        {
            let dx = current_step.x - prev_step.x;
            let dy = current_step.y - prev_step.y;
            let magnitude = (dx * dx + dy * dy).sqrt();
            current_step.direction = if magnitude > f32::EPSILON {
                Some((dx / magnitude, dy / magnitude))
            } else {
                Some((0.0, 0.0))
            };
        }

        let current_trail = self.ensure_current_trail(person_id);
        current_trail.push(current_step.clone());

        let footsteps = self.ensure_history(person_id);

        let mut best_dist = f32::INFINITY;
        for fs in footsteps.iter() {
            let dx = fs.x - x;
            let dy = fs.y - y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < best_dist {
                best_dist = dist;
            }
        }

        if best_dist <= max_dist {
            footsteps.push(current_step.clone());
            return current_step;
        }

        footsteps.push(current_step.clone());
        current_step
    }

    fn prune_older_than(&mut self, max_age: Duration) {
        let now = Instant::now();
        for footsteps in self.history_map.values_mut() {
            footsteps.retain(|step| now.duration_since(step.timestamp) < max_age);
        }
        for trail in self.current_trails.values_mut() {
            trail.retain(|step| now.duration_since(step.timestamp) < max_age);
        }
    }

    fn histories(&self) -> HashMap<usize, Vec<Footstep>> {
        self.history_map.clone()
    }

    /// Remove and return history for a specific person (used when they exit).
    fn take_person(&mut self, person_id: usize) -> Option<Vec<Footstep>> {
        self.current_trails.remove(&person_id);
        self.history_map.remove(&person_id)
    }

    /// Return history only for the provided set of active IDs.
    fn histories_for(&self, active_ids: &HashSet<usize>) -> HashMap<usize, Vec<Footstep>> {
        self.history_map
            .iter()
            .filter_map(|(id, steps)| {
                if active_ids.contains(id) { Some((*id, steps.clone())) } else { None }
            })
            .collect()
    }
}

/// Tracks footsteps for multiple people
pub struct FootstepTracker {
    /// Map from person index to their tracker
    person_trackers: HashMap<usize, PersonFootstepTracker>,
    /// How long to keep footsteps visible (in seconds)
    footstep_display_duration: Duration,
    /// Persistent history with spatial de-duplication
    history: FootstepHistory,
    /// Maximum distance (normalized) to consider two steps the same for history merging
    max_match_distance: f32,
    /// How long to wait before declaring a person has exited after last seen
    exit_timeout: Duration,
    /// Archived footsteps for people who have exited the frame
    archived_histories: Vec<(usize, Vec<Footstep>)>,
}

impl FootstepTracker {
    pub fn new(footstep_display_duration_secs: u64) -> Self {
        Self {
            person_trackers: HashMap::new(),
            footstep_display_duration: Duration::from_secs(footstep_display_duration_secs),
            history: FootstepHistory::new(),
            max_match_distance: 0.03,
            exit_timeout: Duration::from_millis(750),
            archived_histories: Vec::new(),
        }
    }

    /// Update with new frame's keypoints paired with stable IDs
    pub fn update(&mut self, keyed_keypoints: &[(usize, Keypoints)]) -> Vec<FootstepEvent> {
        let now = Instant::now();
        let mut active_ids: HashSet<usize> = HashSet::new();
        let mut new_events = Vec::new();

        for (person_id, person_keypoints) in keyed_keypoints.iter() {
            active_ids.insert(*person_id);

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

            let new_steps = tracker.update(
                (left_ankle[0], left_ankle[1], left_ankle[2]),
                (right_ankle[0], right_ankle[1], right_ankle[2]),
                pelvis
            );

            for step in new_steps {
                let step_with_direction = self.history.match_or_insert(
                    *person_id,
                    step.x,
                    step.y,
                    step.foot,
                    step.timestamp,
                    self.max_match_distance
                );

                new_events.push(FootstepEvent {
                    person_id: *person_id,
                    footstep: step_with_direction,
                });
            }
        }

        // Detect exits: persons not seen this frame and past timeout are archived
        let stale_ids: Vec<usize> = self.person_trackers
            .iter()
            .filter_map(|(id, tracker)| {
                if
                    !active_ids.contains(id) &&
                    now.duration_since(tracker.last_seen) > self.exit_timeout
                {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();

        for id in stale_ids {
            if let Some(steps) = self.history.take_person(id) {
                if !steps.is_empty() {
                    self.archived_histories.push((id, steps));
                }
            }
            self.person_trackers.remove(&id);
        }

        self.history.prune_older_than(self.footstep_display_duration);

        new_events
    }

    /// Get all footsteps grouped by person
    pub fn get_all_footsteps(&self) -> HashMap<usize, Vec<Footstep>> {
        let active_ids: HashSet<usize> = self.person_trackers.keys().cloned().collect();
        self.history.histories_for(&active_ids)
    }

    /// Get archived footsteps for people who have exited the frame.
    pub fn get_archived_footsteps(&self) -> &[(usize, Vec<Footstep>)] {
        &self.archived_histories
    }

    /// Reset all tracking data
    pub fn reset(&mut self) {
        self.person_trackers.clear();
        self.history = FootstepHistory::new();
        self.archived_histories.clear();
    }
}
