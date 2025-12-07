use std::collections::HashMap;
use std::time::Instant;
use crate::footstep_tracker::{Foot, Footstep};


/// A simple manager that holds footsteps per person (person id -> tracker).
#[derive(Debug, Default)]
pub struct FootstepHistory {
    /// Persistent history of detected footsteps per person (older footsteps retained here).
    history_map: HashMap<usize, Vec<Footstep>>,

    /// Current per-frame trail for each person. This is always appended to each time
    /// `match_or_insert` is called, even when a match in history is found.
    current_trails: HashMap<usize, Vec<Footstep>>,
}

/// Result of trying to match a current foot position against history.
#[derive(Debug, Clone)]
pub enum MatchResult {
    /// Found a previous footstep close enough. Contains the index into that person's history and a clone.
    Matched { index: usize, footstep: Footstep },
    /// No match found; a new footstep was inserted at the returned index.
    Inserted { index: usize },
}

impl FootstepHistory {
    pub fn new() -> Self {
        Self { history_map: HashMap::new(), current_trails: HashMap::new() }
    }

    fn ensure_history(&mut self, person_id: usize) -> &mut Vec<Footstep> {
        self.history_map.entry(person_id).or_insert_with(Vec::new)
    }

    fn ensure_current_trail(&mut self, person_id: usize) -> &mut Vec<Footstep> {
        self.current_trails.entry(person_id).or_insert_with(Vec::new)
    }

    /// Find the nearest footstep for `person_id` to (x,y). Returns the index and a clone if found.
    pub fn find_nearest(&self, person_id: usize, x: f32, y: f32, max_dist: f32) -> Option<(usize, Footstep)> {
        let footsteps = self.history_map.get(&person_id)?;
        if footsteps.is_empty() {
            return None;
        }

        let mut best_idx: Option<usize> = None;
        let mut best_dist = f32::INFINITY;

        for (i, fs) in footsteps.iter().enumerate() {
            let dx = fs.x - x;
            let dy = fs.y - y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            if best_dist <= max_dist {
                return Some((idx, footsteps[idx].clone()));
            }
        }
        None
    }

    /// Check history for a close-enough footstep for this person. If one exists (distance <= max_dist)
    /// return MatchResult::Matched with the index and a clone of that footstep. Otherwise insert a new
    /// footstep into the person's history and return MatchResult::Inserted with the new index.
    pub fn match_or_insert(&mut self, person_id: usize, x: f32, y: f32, foot: Foot, timestamp: Instant, max_dist: f32) -> MatchResult {
        // Always append to the current trail for this person
        let current_step = Footstep { x, y, timestamp, foot };
        let current_trail = self.ensure_current_trail(person_id);
        current_trail.push(current_step.clone());

        // Try to find a match in persistent history
        let footsteps = self.ensure_history(person_id);

        let mut best_idx: Option<usize> = None;
        let mut best_dist = f32::INFINITY;

        for (i, fs) in footsteps.iter().enumerate() {
            let dx = fs.x - x;
            let dy = fs.y - y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            if best_dist <= max_dist {
                footsteps.push(current_step);
                return MatchResult::Matched { index: idx, footstep: footsteps[idx].clone() };
            }
        }

        // no match; insert into persistent history
        let new_index = footsteps.len();
        footsteps.push(current_step);
        MatchResult::Inserted { index: new_index }
    }

    /// Get the current trail for a person (most recent entries for the current session/frame sequence).
    pub fn get_current_trail(&self, person_id: usize) -> Option<&[Footstep]> {
        self.current_trails.get(&person_id).map(|v| v.as_slice())
    }

    /// Clear the current trail for a person (e.g., when a new session starts).
    pub fn clear_current_trail(&mut self, person_id: usize) {
        if let Some(v) = self.current_trails.get_mut(&person_id) {
            v.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_match() {
        let mut history = FootstepHistory::new();
        let p = 42usize;
        let t0 = Instant::now();

        // Initially no match; should insert into persistent history and current trail
        match history.match_or_insert(p, 1.0, 2.0, Foot::Left, t0, 0.1) {
            MatchResult::Inserted { index } => assert_eq!(index, 0),
            other => panic!("expected inserted, got {:?}", other),
        }
        let trail = history.get_current_trail(p).expect("current trail should exist");
        assert_eq!(trail.len(), 1);

        // A nearby point within threshold should match
        let t1 = Instant::now();
        match history.match_or_insert(p, 1.05, 2.02, Foot::Left, t1, 0.1) {
            MatchResult::Matched { index, footstep } => {
                assert_eq!(index, 0);
                assert_eq!(footstep.x, 1.0);
                assert_eq!(footstep.y, 2.0);
            }
            other => panic!("expected match, got {:?}", other),
        }
        // Even though it matched existing history, current trail should be appended
        let trail = history.get_current_trail(p).expect("current trail should exist");
        assert_eq!(trail.len(), 2);

        // A far point should insert a second footstep
        let t2 = Instant::now();
        match history.match_or_insert(p, 10.0, 10.0, Foot::Right, t2, 0.5) {
            MatchResult::Inserted { index } => assert_eq!(index, 1),
            other => panic!("expected inserted, got {:?}", other),
        }
        let trail = history.get_current_trail(p).expect("current trail should exist");
        assert_eq!(trail.len(), 3);
    }
}
