use anyhow::Result;
use opencv::{ core, imgproc, prelude::* };
use coreml_rs_fork::{ CoreMLModelOptions, CoreMLModelWithState, ComputePlatform };
use std::collections::HashMap;

#[cfg(feature = "debug")]
use tracing::info;

/// Person class ID
const PERSON_CLASS_ID: usize = 0;

/// Bounding box
#[derive(Debug, Clone)]
pub struct BoundingBox {
    // normalized coordinates (0-1)
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
}

impl BoundingBox {
    /// To pixels
    /// YOLO format: center x,y w,h
    pub fn to_pixels(&self, img_width: i32, img_height: i32) -> (i32, i32, i32, i32) {
        let w = (self.width * (img_width as f32)) as i32;
        let h = (self.height * (img_height as f32)) as i32;
        let x = (self.x * (img_width as f32) - (w as f32) / 2.0) as i32;
        let y = (self.y * (img_height as f32) - (h as f32) / 2.0) as i32;
        (x, y, w, h)
    }

    /// Expand box
    /// Center format
    pub fn expand(&self, factor: f32) -> Self {
        let new_width = (self.width * factor).min(1.0);
        let new_height = (self.height * factor).min(1.0);

        // clamp to keep in bounds
        let x = self.x.clamp(new_width / 2.0, 1.0 - new_width / 2.0);
        let y = self.y.clamp(new_height / 2.0, 1.0 - new_height / 2.0);

        Self {
            x,
            y,
            width: new_width,
            height: new_height,
            confidence: self.confidence,
        }
    }

    pub fn iou(&self, other: &Self) -> f32 {
        let (x1, y1, w1, h1) = (
            self.x - self.width * 0.5,
            self.y - self.height * 0.5,
            self.width,
            self.height,
        );
        let (x2, y2, w2, h2) = (
            other.x - other.width * 0.5,
            other.y - other.height * 0.5,
            other.width,
            other.height,
        );

        let inter_x1 = x1.max(x2);
        let inter_y1 = y1.max(y2);
        let inter_x2 = (x1 + w1).min(x2 + w2);
        let inter_y2 = (y1 + h1).min(y2 + h2);

        let inter_w = (inter_x2 - inter_x1).max(0.0);
        let inter_h = (inter_y2 - inter_y1).max(0.0);
        let inter_area = inter_w * inter_h;

        let union_area = w1 * h1 + w2 * h2 - inter_area;
        if union_area <= 0.0 {
            0.0
        } else {
            inter_area / union_area
        }
    }

    pub fn center_distance2(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    pub fn corner_l2_avg(&self, other: &Self) -> f32 {
        let half_w1 = self.width * 0.5;
        let half_h1 = self.height * 0.5;
        let half_w2 = other.width * 0.5;
        let half_h2 = other.height * 0.5;

        let corners1 = [
            (self.x - half_w1, self.y - half_h1),
            (self.x + half_w1, self.y - half_h1),
            (self.x - half_w1, self.y + half_h1),
            (self.x + half_w1, self.y + half_h1),
        ];

        let corners2 = [
            (other.x - half_w2, other.y - half_h2),
            (other.x + half_w2, other.y - half_h2),
            (other.x - half_w2, other.y + half_h2),
            (other.x + half_w2, other.y + half_h2),
        ];

        let mut acc = 0.0;
        for i in 0..4 {
            let dx = corners1[i].0 - corners2[i].0;
            let dy = corners1[i].1 - corners2[i].1;
            acc += dx * dx + dy * dy;
        }
        acc * 0.25
    }
}

/// YOLO config
#[derive(Debug, Clone)]
pub struct YoloDetectorConfig {
    pub model_path: String,
    pub confidence_threshold: f32,
    pub input_size: usize,
}

impl Default for YoloDetectorConfig {
    fn default() -> Self {
        Self {
            model_path: "models/yolo11n.mlpackage".to_string(),
            confidence_threshold: 0.1,
            input_size: 640,
        }
    }
}

/// YOLO detector
pub struct YoloDetector {
    config: YoloDetectorConfig,
    model: CoreMLModelWithState,
}

#[derive(Debug, Clone)]
struct Track {
    id: usize,
    bbox: BoundingBox,
    age: usize,
    velocity: (f32, f32),
    last_center: (f32, f32),
}

/// Person tracker
pub struct PersonTracker {
    tracks: HashMap<usize, Track>,
    next_id: usize,
    max_age: usize,
}

impl YoloDetector {
    /// New YOLO detector
    pub fn new(config: YoloDetectorConfig) -> Result<Self> {
        #[cfg(feature = "debug")]
        info!("Initializing YOLO Detector...");

        let mut model_options = CoreMLModelOptions::default();
        model_options.compute_platform = ComputePlatform::CpuAndANE;

        let model = CoreMLModelWithState::new(&config.model_path, model_options)
            .load()
            .map_err(|e| anyhow::anyhow!("Failed to load YOLO model: {:?}", e))?;

        #[cfg(feature = "debug")]
        {
            info!("✓ YOLO model loaded with Neural Engine acceleration!");
            info!("Model: {}", config.model_path);
            info!("Input size: {}x{}", config.input_size, config.input_size);
        }

        Ok(Self { config, model })
    }

    /// Detect people
    pub fn detect_people(&mut self, frame: &Mat) -> Result<Vec<BoundingBox>> {
        // convert frame to 640x640 BGRA
        let input_bgra = self.preprocess_frame(frame)?;

        // Add CVPixelBuffer input
        self.model
            .add_input_cvpixelbuffer(
                "image",
                self.config.input_size,
                self.config.input_size,
                input_bgra
            )
            .map_err(|e| anyhow::anyhow!("Failed to add YOLO image input: {:?}", e))?;

        // TODO: using model defaults for iouThreshold and confidenceThreshold

        // Run inference
        let mut outputs = self.model
            .predict()
            .map_err(|e| anyhow::anyhow!("YOLO prediction failed: {:?}", e))?;

        // Extract outputs
        let confidence_array = outputs.outputs
            .remove("confidence")
            .ok_or_else(|| anyhow::anyhow!("Missing 'confidence' output"))?;

        let coordinates_array = outputs.outputs
            .remove("coordinates")
            .ok_or_else(|| anyhow::anyhow!("Missing 'coordinates' output"))?;

        let confidence_tensor = confidence_array.extract_to_tensor::<f32>();
        let confidence_shape = confidence_tensor.shape().to_vec();
        let (confidence_data, _) = confidence_tensor.into_raw_vec_and_offset();

        let coordinates_tensor = coordinates_array.extract_to_tensor::<f32>();
        let (coordinates_data, _) = coordinates_tensor.into_raw_vec_and_offset();

        let num_detections = if confidence_shape.len() >= 2 {
            confidence_shape[0]
        } else {
            confidence_data.len() / 80
        };

        let mut people = Vec::new();

        for i in 0..num_detections {
            let person_conf = confidence_data[i * 80 + PERSON_CLASS_ID];

            if person_conf >= self.config.confidence_threshold {
                let coord_offset = i * 4;
                let bbox = BoundingBox {
                    x: coordinates_data[coord_offset + 0],
                    y: coordinates_data[coord_offset + 1],
                    width: coordinates_data[coord_offset + 2],
                    height: coordinates_data[coord_offset + 3],
                    confidence: person_conf,
                };
                people.push(bbox);
            }
        }

        Ok(people)
    }

    /// Convert frame to 640x640 BGRA
    fn preprocess_frame(&self, frame: &Mat) -> Result<Vec<u8>> {
        let mut resized = Mat::default();
        let size = core::Size::new(self.config.input_size as i32, self.config.input_size as i32);

        imgproc::resize(frame, &mut resized, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;

        // Convert BGR to BGRA
        let mut bgra = Mat::default();
        imgproc::cvt_color(
            &resized,
            &mut bgra,
            imgproc::COLOR_BGR2BGRA,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT
        )?;

        let bgra_data = bgra.data_bytes()?.to_vec();

        Ok(bgra_data)
    }

    /// Crop a region from the frame
    pub fn crop_region(frame: &Mat, bbox: &BoundingBox, target_size: usize) -> Result<Mat> {
        let frame_height = frame.rows();
        let frame_width = frame.cols();

        let (x, y, w, h) = bbox.to_pixels(frame_width, frame_height);

        let x = x.max(0).min(frame_width - 1);
        let y = y.max(0).min(frame_height - 1);
        let w = w.max(1).min(frame_width - x);
        let h = h.max(1).min(frame_height - y);

        let roi = core::Rect::new(x, y, w, h);
        let cropped = Mat::roi(frame, roi)?;

        let mut resized = Mat::default();
        let size = core::Size::new(target_size as i32, target_size as i32);
        imgproc::resize(&cropped, &mut resized, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;

        Ok(resized)
    }
}

impl PersonTracker {
    pub fn new(max_age: usize) -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 0,
            max_age,
        }
    }

    pub fn assign_ids(&mut self, detections: Vec<BoundingBox>) -> Vec<(usize, BoundingBox)> {
        let mut results = Vec::with_capacity(detections.len());

        for track in self.tracks.values_mut() {
            track.age += 1;
        }

        let dets: Vec<(usize, BoundingBox, (f32, f32), f32)> = detections
            .into_iter()
            .enumerate()
            .map(|(idx, det)| {
                let center = (det.x, det.y);
                let diag = (det.width * det.width + det.height * det.height).sqrt();
                (idx, det, center, diag)
            })
            .collect();

        let mut det_assigned: Vec<Option<usize>> = vec![None; dets.len()];
        let mut track_used = std::collections::HashSet::new();

        let mut candidates: Vec<(f32, usize, usize, f32, f32)> = Vec::new();
        for (track_id, track) in self.tracks.iter() {
            let pred_center = (
                track.last_center.0 + track.velocity.0,
                track.last_center.1 + track.velocity.1,
            );

            let track_diag = (
                track.bbox.width * track.bbox.width +
                track.bbox.height * track.bbox.height
            ).sqrt();

            for (det_idx, det, det_center, det_diag) in dets.iter() {
                let dx = det_center.0 - pred_center.0;
                let dy = det_center.1 - pred_center.1;
                let center_dist = (dx * dx + dy * dy).sqrt();

                let avg_diag = (det_diag + track_diag) * 0.5;
                let gate = (avg_diag * 0.9 + 0.03 * (1.0 + (track.age as f32) * 0.2)).max(0.06);
                let iou = det.iou(&track.bbox);

                if center_dist > gate && iou <= 0.12 {
                    continue;
                }

                let cost = center_dist * 3.0 - iou * 2.0 + (track.age as f32) * 0.03;
                candidates.push((cost, *track_id, *det_idx, center_dist, iou));
            }
        }
        // Match to older ids
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for (cost, track_id, det_idx, _dist, _iou) in candidates {
            if track_used.contains(&track_id) || det_assigned[det_idx].is_some() {
                continue;
            }
            if cost > 0.8 {
                continue;
            }
            track_used.insert(track_id);
            det_assigned[det_idx] = Some(track_id);
        }

        for (det_idx, _det, det_center, det_diag) in dets.iter() {
            if det_assigned[*det_idx].is_some() {
                continue;
            }

            let mut best: Option<(usize, f32)> = None; // (track_id, dist)
            for (id, track) in self.tracks.iter() {
                if track_used.contains(id) || track.age > 3 {
                    continue;
                }

                let pred_center = (
                    track.last_center.0 + track.velocity.0,
                    track.last_center.1 + track.velocity.1,
                );
                let dx = det_center.0 - pred_center.0;
                let dy = det_center.1 - pred_center.1;
                let center_dist = (dx * dx + dy * dy).sqrt();

                let track_diag = (
                    track.bbox.width * track.bbox.width +
                    track.bbox.height * track.bbox.height
                ).sqrt();
                let reuse_gate = ((det_diag + track_diag) * 0.5 * 1.3).max(0.08);

                if center_dist <= reuse_gate {
                    match best {
                        Some((_, best_dist)) => {
                            if center_dist < best_dist {
                                best = Some((*id, center_dist));
                            }
                        }
                        None => {
                            best = Some((*id, center_dist));
                        }
                    }
                }
            }

            if let Some((id, _)) = best {
                track_used.insert(id);
                det_assigned[*det_idx] = Some(id);
            } else {
                let id = self.next_id;
                self.next_id += 1;
                det_assigned[*det_idx] = Some(id);
            }
        }

        for (det_idx, det, det_center, _det_diag) in dets.into_iter() {
            let id = det_assigned[det_idx].expect("every detection should have an id");

            let new_velocity = if let Some(prev) = self.tracks.get(&id) {
                let prev_center = prev.last_center;
                let vx = (det_center.0 - prev_center.0) * 0.6 + prev.velocity.0 * 0.4;
                let vy = (det_center.1 - prev_center.1) * 0.6 + prev.velocity.1 * 0.4;
                (vx, vy)
            } else {
                (0.0, 0.0)
            };

            self.tracks.insert(id, Track {
                id,
                bbox: det.clone(),
                age: 0,
                velocity: new_velocity,
                last_center: det_center,
            });

            results.push((id, det));
        }

        // Drop stale trails
        self.tracks.retain(|_, track| track.age <= self.max_age);

        results
    }
}
