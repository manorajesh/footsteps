use crate::pose_detector::{ Keypoint, Keypoints };
use anyhow::{ Context, Result };
use std::collections::{ HashMap, HashSet };
use std::fs;
use std::path::Path;
use std::time::{ Duration, Instant };
use std::time::{ SystemTime, UNIX_EPOCH };
use serde::{ Deserialize, Serialize };

/// Footstep
#[derive(Debug, Clone)]
pub struct Footstep {
    pub x: f32,
    pub y: f32,
    /// Direction
    pub direction: Option<(f32, f32)>,
    pub timestamp: Instant,
    pub foot: Foot,
}

/// Footstep event
#[derive(Debug, Clone)]
pub struct FootstepEvent {
    pub person_id: usize,
    pub footstep: Footstep,
    pub history: Vec<Footstep>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Serialize, Deserialize)]
pub enum Foot {
    Left,
    Right,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PersistedFootstep {
    x: f32,
    y: f32,
    direction: Option<(f32, f32)>,
    foot: Foot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedPath {
    pub timestamp_ms: u64,
    pub steps: Vec<PersistedFootstep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedHistoryStore {
    version: u32,
    saved_at_unix_ms: u64,
    histories: Vec<PersistedPath>,
}

const MIN_CONFIDENCE: f32 = 0.2;
const HISTORY_MATCH_DISTANCE: f32 = 0.02;
const HISTORY_DIRECTION_COS_THRESHOLD: f32 = 0.1;
const HISTORY_DIRECTION_WEIGHT: f32 = 0.4;
const MAX_PAST_HISTORIES: usize = 5000;
const MAX_ARCHIVED_HISTORIES: usize = 2000;
const MAX_STEPS_PER_HISTORY: usize = 300;
const MAX_HISTORY_FILE_BYTES: usize = 1024 * 1024 * 1024;

#[derive(Debug, Default)]
struct FootstepHistory {
    history_map: HashMap<usize, Vec<Footstep>>,
    current_trails: HashMap<usize, Vec<Footstep>>,
}

#[derive(Debug, Clone, Copy)]
struct EmaPoint {
    pos: Option<(f32, f32)>,
    alpha: f32,
}

impl EmaPoint {
    fn new(alpha: f32) -> Self {
        Self { pos: None, alpha }
    }

    fn update(&mut self, new_pos: (f32, f32)) -> (f32, f32) {
        if let Some(current) = self.pos {
            let updated = (
                current.0 * (1.0 - self.alpha) + new_pos.0 * self.alpha,
                current.1 * (1.0 - self.alpha) + new_pos.1 * self.alpha,
            );
            self.pos = Some(updated);
            updated
        } else {
            self.pos = Some(new_pos);
            new_pos
        }
    }
}

/// Person footstep tracker
#[derive(Debug)]
struct PersonFootstepTracker {
    ema_pelvis: EmaPoint,
    ema_velocity: EmaPoint,
    
    last_seen: Instant,
    last_pelvis: Option<(f32, f32)>,
    last_valid_dir: Option<(f32, f32)>,
    
    // Distance-based synthetic stride tracking
    last_step_pos: Option<(f32, f32)>,
    next_foot_is_left: bool,
    last_step_time: Instant,
}

impl PersonFootstepTracker {
    fn new() -> Self {
        Self {
            ema_pelvis: EmaPoint::new(0.3),
            ema_velocity: EmaPoint::new(0.1),
            
            last_seen: Instant::now(),
            last_pelvis: None,
            last_valid_dir: None,
            
            last_step_pos: None,
            next_foot_is_left: true,
            last_step_time: Instant::now(),
        }
    }

    fn update_motion_hint(&mut self, pelvis: (f32, f32, f32)) -> ((f32, f32), Option<(f32, f32)>) {
        let raw_pelvis_pos = (pelvis.0, pelvis.1);
        
        if pelvis.2 < MIN_CONFIDENCE {
            return (self.ema_pelvis.pos.unwrap_or(raw_pelvis_pos), self.last_valid_dir);
        }
        
        let smoothed_pelvis = self.ema_pelvis.update(raw_pelvis_pos);

        if let Some(prev) = self.last_pelvis {
            let dx = smoothed_pelvis.0 - prev.0;
            let dy = smoothed_pelvis.1 - prev.1;
            
            // Calculate and smooth the raw velocity vector instead of normalizing noisy micro-movements
            let smoothed_vel = self.ema_velocity.update((dx, dy));
            let speed = (smoothed_vel.0 * smoothed_vel.0 + smoothed_vel.1 * smoothed_vel.1).sqrt();
            
            // Only update the direction if we are actually moving significantly (e.g. out of pose jitter)
            if speed > 0.0005 {
                self.last_valid_dir = Some((smoothed_vel.0 / speed, smoothed_vel.1 / speed));
            }
        }

        self.last_pelvis = Some(smoothed_pelvis);
        (smoothed_pelvis, self.last_valid_dir)
    }

    fn update(
        &mut self,
        _left_ankle: (f32, f32, f32),
        _right_ankle: (f32, f32, f32),
        motion_dir: Option<(f32, f32)>
    ) -> Vec<Footstep> {
        let mut new_steps = Vec::new();
        self.last_seen = Instant::now();
        
        let pelvis = self.ema_pelvis.pos.unwrap_or((0.0, 0.0));
        let fixed_time = Instant::now();
        let time_since_last = fixed_time.duration_since(self.last_step_time).as_secs_f32();

        if let Some(dir) = motion_dir {
            // We use standard distance measurement to generate smooth fixed footsteps 
            // completely ignoring the exact ankle positioning that caused the jitter
            let stride_length = 0.04; // The required distance traveled before step
            let stride_width = 0.015; // The orthogonal distance outward

            // Setup first step
            if self.last_step_pos.is_none() {
                self.last_step_pos = Some((pelvis.0, pelvis.1));
                self.last_step_time = fixed_time;
                return new_steps;
            }

            let last_pos = self.last_step_pos.unwrap();
            let dx = pelvis.0 - last_pos.0;
            let dy = pelvis.1 - last_pos.1;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist >= stride_length && time_since_last > 0.3 {
                // Orthogonal vector (rotate by 90/-90 deg) depending on left/right foot
                let perp_dir = if self.next_foot_is_left {
                    (-dir.1, dir.0) // Left side
                } else {
                    (dir.1, -dir.0) // Right side
                };

                let foot_x = pelvis.0 + perp_dir.0 * stride_width;
                let foot_y = pelvis.1 + perp_dir.1 * stride_width;

                new_steps.push(Footstep {
                    x: foot_y, // Note the coordinate flip x <-> y in existing logic
                    y: foot_x,
                    direction: motion_dir,
                    timestamp: fixed_time,
                    foot: if self.next_foot_is_left { Foot::Left } else { Foot::Right },
                });

                self.last_step_pos = Some((pelvis.0, pelvis.1));
                self.next_foot_is_left = !self.next_foot_is_left;
                self.last_step_time = fixed_time;
            }
        } else {
            // Drop tracking if stationary over 0.5s so they can start fresh
            if time_since_last > 0.5 {
                self.last_step_pos = None;
            }
        }

        new_steps
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

    fn seed_history(&mut self, person_id: usize, history: Vec<Footstep>) {
        if history.is_empty() {
            return;
        }

        let entry = self.history_map.entry(person_id).or_insert_with(Vec::new);
        if entry.is_empty() {
            *entry = history.clone();
        } else {
            let mut merged = history.clone();
            merged.extend(entry.drain(..));
            *entry = merged;
        }

        let trail = self.current_trails.entry(person_id).or_insert_with(Vec::new);
        if trail.is_empty() {
            *trail = history;
        }
    }

    #[allow(dead_code)]
    fn histories(&self) -> HashMap<usize, Vec<Footstep>> {
        self.history_map.clone()
    }

    /// Take person history
    fn take_person(&mut self, person_id: usize) -> Option<Vec<Footstep>> {
        self.current_trails.remove(&person_id);
        self.history_map.remove(&person_id)
    }

    /// Histories for active IDs
    fn histories_for(&self, active_ids: &HashSet<usize>) -> HashMap<usize, Vec<Footstep>> {
        self.history_map
            .iter()
            .filter_map(|(id, steps)| {
                if active_ids.contains(id) { Some((*id, steps.clone())) } else { None }
            })
            .collect()
    }
}

/// Footstep tracker
pub struct FootstepTracker {
    /// Person trackers
    person_trackers: HashMap<usize, PersonFootstepTracker>,
    /// Display duration
    footstep_display_duration: Duration,
    /// History
    history: FootstepHistory,
    /// Merge distance
    max_match_distance: f32,
    /// Exit timeout
    exit_timeout: Duration,
    /// Archived footsteps
    archived_histories: Vec<(usize, Vec<Footstep>)>,
    /// Active to archived
    archived_matches: HashMap<usize, usize>,
    /// Permanent store of all past trajectories
    pub past_histories: Vec<(u64, Vec<Footstep>)>,
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
            archived_matches: HashMap::new(),
            past_histories: Vec::new(),
        }
    }

    fn last_pose_hint(steps: &[Footstep]) -> Option<((f32, f32), Option<(f32, f32)>)> {
        let last = steps.last()?;
        let last_pos = (last.x, last.y);

        let dir = if let Some(dir) = last.direction {
            Some(dir)
        } else if steps.len() >= 2 {
            let prev = &steps[steps.len() - 2];
            let dx = last.x - prev.x;
            let dy = last.y - prev.y;
            let mag = (dx * dx + dy * dy).sqrt();
            if mag > f32::EPSILON {
                Some((dx / mag, dy / mag))
            } else {
                None
            }
        } else {
            None
        };

        Some((last_pos, dir))
    }

    fn try_match_archived_history(
        &mut self,
        person_id: usize,
        pelvis_pos: (f32, f32),
        motion_dir: Option<(f32, f32)>
    ) {
        if self.archived_matches.contains_key(&person_id) {
            return;
        }

        let mut best: Option<(usize, f32, usize)> = None; // (arch_idx, cost, archived_id)

        for (idx, (arch_id, steps)) in self.archived_histories.iter().enumerate() {
            if let Some((last_pos, hist_dir)) = Self::last_pose_hint(steps) {
                let dx = last_pos.0 - pelvis_pos.0;
                let dy = last_pos.1 - pelvis_pos.1;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > HISTORY_MATCH_DISTANCE {
                    continue;
                }

                let dir_penalty = match (hist_dir, motion_dir) {
                    (Some(hd), Some(cd)) => {
                        let dot = (hd.0 * cd.0 + hd.1 * cd.1).clamp(-1.0, 1.0);
                        if dot < HISTORY_DIRECTION_COS_THRESHOLD {
                            continue;
                        }
                        1.0 - dot
                    }
                    _ => 0.0,
                };

                let cost = dist + dir_penalty * HISTORY_DIRECTION_WEIGHT;
                if best.map_or(true, |(_, c, _)| cost < c) {
                    best = Some((idx, cost, *arch_id));
                }
            }
        }

        if let Some((idx, _, arch_id)) = best {
            let (_, steps) = self.archived_histories.remove(idx);
            self.history.seed_history(person_id, steps);
            self.archived_matches.insert(person_id, arch_id);
        }
    }

    /// Update tracker with new keypoints
    pub fn update(&mut self, keyed_keypoints: &[(usize, Keypoints)]) -> Vec<FootstepEvent> {
        let now = Instant::now();
        let mut active_ids: HashSet<usize> = HashSet::new();
        let mut new_events = Vec::new();

        for (person_id, person_keypoints) in keyed_keypoints.iter() {
            active_ids.insert(*person_id);

            let left_ankle = person_keypoints[Keypoint::LeftAnkle as usize];
            let right_ankle = person_keypoints[Keypoint::RightAnkle as usize];
            let left_hip = person_keypoints[Keypoint::LeftHip as usize];
            let right_hip = person_keypoints[Keypoint::RightHip as usize];

            let pelvis = (
                (left_hip[0] + right_hip[0]) * 0.5,
                (left_hip[1] + right_hip[1]) * 0.5,
                (left_hip[2] + right_hip[2]) * 0.5,
            );

            let (pelvis_pos, motion_dir) = {
                let tracker = self.person_trackers
                    .entry(*person_id)
                    .or_insert_with(PersonFootstepTracker::new);
                tracker.update_motion_hint(pelvis)
            };

            self.try_match_archived_history(*person_id, pelvis_pos, motion_dir);

            let tracker = self.person_trackers
                .get_mut(person_id)
                .expect("tracker should exist after insertion");

            let new_steps = tracker.update(
                (left_ankle[0], left_ankle[1], left_ankle[2]),
                (right_ankle[0], right_ankle[1], right_ankle[2]),
                motion_dir
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

                let history = self.history.histories_for(&[*person_id].into())
                    .remove(person_id)
                    .unwrap_or_default();

                new_events.push(FootstepEvent {
                    person_id: *person_id,
                    footstep: step_with_direction,
                    history,
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

        // Archive stale ids
        for id in stale_ids {
            let was_archived_match = self.archived_matches.remove(&id).is_some();

            if let Some(steps) = self.history.take_person(id) {
                if !steps.is_empty() {
                    if !was_archived_match {
                        self.archived_histories.push((id, steps.clone()));
                    }
                    let now_ms = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64;
                    self.past_histories.push((now_ms, steps));
                }
            }

            self.person_trackers.remove(&id);
        }

        // Drop any archived match links for people not present in the current frame
        self.archived_matches.retain(|id, _| active_ids.contains(id));

        self.history.prune_older_than(self.footstep_display_duration);
        self.enforce_storage_limits();

        new_events
    }

    pub fn get_all_footsteps(&self) -> HashMap<usize, Vec<Footstep>> {
        let active_ids: HashSet<usize> = self.person_trackers.keys().cloned().collect();
        self.history.histories_for(&active_ids)
    }
    /// Checks if any currently active paths share consecutive points with historical paths.
    /// Returns a list of active person IDs mapped to the matched historical paths.
    pub fn get_matched_past_paths(&self) -> HashMap<usize, Vec<Footstep>> {
        let min_steps = 5; // N: minimum length to consider for a match
        let distance_threshold = 0.05; // Maximum distance to consider a step "matching"
        
        // Grab paths for all currently active people
        let current_histories = self.get_all_footsteps();
        let mut active_entries: Vec<_> = current_histories.into_iter().collect();
        // Sort by ID to ensure deterministic matching order (prevent flickering if people cross paths)
        active_entries.sort_by_key(|(id, _)| *id);
        
        let mut matches = HashMap::new();
        let mut used_past_indices = HashSet::new();

        for (person_id, current_steps) in active_entries {
            if current_steps.len() < min_steps {
                continue;
            }

            // We only need to check the last N steps of their current path
            let active_segment = &current_steps[current_steps.len() - min_steps..];

            // Compare against every accumulated past history
            'history_loop: for (past_idx, (_, past_path)) in self.past_histories.iter().enumerate() {
                if used_past_indices.contains(&past_idx) {
                    continue;
                }
                
                if past_path.len() < min_steps {
                    continue;
                }

                // Slide a window over the old path
                let windows_to_check = past_path.len() - min_steps + 1;
                for i in 0..windows_to_check {
                    let past_segment = &past_path[i..i + min_steps];
                    
                    let mut is_match = true;
                    for (active_step, past_step) in active_segment.iter().zip(past_segment.iter()) {
                        let dx = active_step.x - past_step.x;
                        let dy = active_step.y - past_step.y;
                        let dist = (dx * dx + dy * dy).sqrt();
                        
                        if dist > distance_threshold {
                            is_match = false;
                            break;
                        }
                    }

                    if is_match {
                        matches.insert(person_id, past_path.clone());
                        used_past_indices.insert(past_idx);
                        break 'history_loop;
                    }
                }
            }
        }

        matches
    }
    pub fn get_archived_footsteps(&self) -> &[(usize, Vec<Footstep>)] {
        &self.archived_histories
    }

    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.person_trackers.clear();
        self.history = FootstepHistory::new();
        self.archived_histories.clear();
        self.archived_matches.clear();
        self.past_histories.clear();
    }

    pub fn past_history_count(&self) -> usize {
        self.past_histories.len()
    }

    pub fn load_past_histories<P: AsRef<Path>>(&mut self, path: P) -> Result<usize> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(0);
        }

        let raw = fs::read_to_string(path).with_context(|| {
            format!("Failed to read footstep history file: {}", path.display())
        })?;

        let store: PersistedHistoryStore = serde_json::from_str(&raw).with_context(|| {
            format!("Failed to parse footstep history JSON: {}", path.display())
        })?;

        let now = Instant::now();
        self.past_histories = store.histories
            .into_iter()
            .map(|persisted_path| {
                #[cfg(feature = "debug")]
                tracing::info!("Loaded history path with timestamp: {}", persisted_path.timestamp_ms);
                let steps = persisted_path.steps
                    .into_iter()
                    .map(|step| Footstep {
                        x: step.x,
                        y: step.y,
                        direction: step.direction,
                        timestamp: now,
                        foot: step.foot,
                    })
                    .collect::<Vec<_>>();
                (persisted_path.timestamp_ms, steps)
            })
            .collect();

        self.enforce_storage_limits();

        Ok(self.past_histories.len())
    }

    pub fn save_past_histories<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create history directory: {}", parent.display())
            })?;
        }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut store = PersistedHistoryStore {
            version: 1,
            saved_at_unix_ms: now_ms,
            histories: self.past_histories
                .iter()
                .map(|(timestamp_ms, path_steps)| {
                    #[cfg(feature = "debug")]
                    tracing::debug!("Saving history path with timestamp: {}", timestamp_ms);
                    let start = path_steps.len().saturating_sub(MAX_STEPS_PER_HISTORY);
                    let steps = path_steps[start..]
                        .iter()
                        .map(|step| PersistedFootstep {
                            x: step.x,
                            y: step.y,
                            direction: step.direction,
                            foot: step.foot,
                        })
                        .collect::<Vec<_>>();
                    PersistedPath {
                        timestamp_ms: *timestamp_ms,
                        steps,
                    }
                })
                .collect(),
        };

        // Keep persisted history bounded so long-running sessions do not create huge files.
        while store.histories.len() > MAX_PAST_HISTORIES {
            store.histories.remove(0);
        }

        let tmp_path = path.with_extension("json.tmp");
        let mut payload = serde_json::to_string_pretty(&store).context(
            "Failed to serialize footstep history to JSON"
        )?;

        while payload.len() > MAX_HISTORY_FILE_BYTES {
            if store.histories.len() > 1 {
                store.histories.remove(0);
            } else if let Some(first) = store.histories.first_mut() {
                if first.steps.len() <= 1 {
                    break;
                }
                let drop_count = (first.steps.len() / 10).max(1);
                first.steps.drain(0..drop_count.min(first.steps.len() - 1));
            } else {
                break;
            }

            payload = serde_json::to_string_pretty(&store).context(
                "Failed to serialize footstep history to JSON"
            )?;
        }

        fs::write(&tmp_path, payload).with_context(|| {
            format!("Failed to write temporary footstep history file: {}", tmp_path.display())
        })?;

        fs::rename(&tmp_path, path).with_context(|| {
            format!(
                "Failed to move temporary history file into place: {} -> {}",
                tmp_path.display(),
                path.display()
            )
        })?;

        Ok(())
    }

    fn enforce_storage_limits(&mut self) {
        for (_, steps) in self.past_histories.iter_mut() {
            if steps.len() > MAX_STEPS_PER_HISTORY {
                let keep_from = steps.len() - MAX_STEPS_PER_HISTORY;
                steps.drain(0..keep_from);
            }
        }
        self.past_histories.retain(|(_, steps)| !steps.is_empty());
        if self.past_histories.len() > MAX_PAST_HISTORIES {
            let drop_count = self.past_histories.len() - MAX_PAST_HISTORIES;
            self.past_histories.drain(0..drop_count);
        }

        for (_, steps) in self.archived_histories.iter_mut() {
            if steps.len() > MAX_STEPS_PER_HISTORY {
                let keep_from = steps.len() - MAX_STEPS_PER_HISTORY;
                steps.drain(0..keep_from);
            }
        }
        self.archived_histories.retain(|(_, steps)| !steps.is_empty());
        if self.archived_histories.len() > MAX_ARCHIVED_HISTORIES {
            let drop_count = self.archived_histories.len() - MAX_ARCHIVED_HISTORIES;
            self.archived_histories.drain(0..drop_count);
        }
    }
}
