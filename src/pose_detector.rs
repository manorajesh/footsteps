use anyhow::Result;
use opencv::{ core, imgproc, prelude::* };
use coreml_rs::{ CoreMLModelOptions, CoreMLModelWithState, ComputePlatform };
use ndarray::Array4;
use half;

/// Keypoint indices for the 17-point COCO skeleton
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum Keypoint {
    Nose = 0,
    LeftEye = 1,
    RightEye = 2,
    LeftEar = 3,
    RightEar = 4,
    LeftShoulder = 5,
    RightShoulder = 6,
    LeftElbow = 7,
    RightElbow = 8,
    LeftWrist = 9,
    RightWrist = 10,
    LeftHip = 11,
    RightHip = 12,
    LeftKnee = 13,
    RightKnee = 14,
    LeftAnkle = 15,
    RightAnkle = 16,
}

const CONFIDENCE_THRESHOLD: f32 = 0.2;

/// Each keypoint contains (y, x, confidence)
pub type KeypointData = [f32; 3];
pub type Keypoints = Vec<KeypointData>;
pub type MultiPoseKeypoints = Vec<Keypoints>;

/// Configuration for the pose detector
#[derive(Debug, Clone)]
pub struct PoseDetectorConfig {
    pub model_path: String,
    pub model_input_size: usize,
    pub num_keypoints: usize,
    pub max_people: usize,
    /// Confidence threshold below which we use temporal smoothing more aggressively
    pub low_confidence_threshold: f32,
    /// Smoothing factor for exponential moving average (0.0 = use only current, 1.0 = use only previous)
    pub smoothing_alpha: f32,
    /// Smoothing factor when confidence is low (higher = more smoothing)
    pub low_confidence_alpha: f32,
}

impl Default for PoseDetectorConfig {
    fn default() -> Self {
        Self {
            model_path: "models/movenet_singlepose_lightning.mlpackage".to_string(),
            model_input_size: 256,
            num_keypoints: 17,
            max_people: 1,
            low_confidence_threshold: 0.5,
            smoothing_alpha: 0.3,
            low_confidence_alpha: 0.7,
        }
    }
}

/// Pose detector using CoreML with temporal smoothing
pub struct PoseDetector {
    config: PoseDetectorConfig,
    model: CoreMLModelWithState,
    /// Previous frame's keypoints for temporal smoothing
    previous_keypoints: Option<MultiPoseKeypoints>,
    #[cfg(feature = "debug")]
    pub frame_counter: usize,
}

impl PoseDetector {
    /// Create a new pose detector with the given configuration
    pub fn new(config: PoseDetectorConfig) -> Result<Self> {
        #[cfg(feature = "debug")]
        println!("Initializing Pose Detector with CoreML...");

        let mut model_options = CoreMLModelOptions::default();
        model_options.compute_platform = ComputePlatform::CpuAndANE;

        let model = CoreMLModelWithState::new(&config.model_path, model_options)
            .load()
            .map_err(|e| anyhow::anyhow!("Failed to load CoreML model: {:?}", e))?;

        #[cfg(feature = "debug")]
        {
            println!("✓ CoreML model loaded with Neural Engine acceleration!");
            println!("Model: {}", config.model_path);
            println!("Input size: {}", config.model_input_size);
        }

        #[cfg(feature = "debug")]
        println!("✓ Pose Detector ready with temporal smoothing!");

        Ok(Self {
            config,
            model,
            previous_keypoints: None,
            #[cfg(feature = "debug")]
            frame_counter: 0,
        })
    }

    /// Preprocess a frame for inference
    fn preprocess_frame(&self, frame: &Mat) -> Result<Mat> {
        let mut resized = Mat::default();
        let size = core::Size::new(
            self.config.model_input_size as i32,
            self.config.model_input_size as i32
        );

        imgproc::resize(frame, &mut resized, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;

        let mut rgb = Mat::default();
        imgproc::cvt_color(
            &resized,
            &mut rgb,
            imgproc::COLOR_BGR2RGB,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT
        )?;

        Ok(rgb)
    }

    /// Detect poses in a frame
    pub fn detect_pose(&mut self, frame: &Mat) -> Result<MultiPoseKeypoints> {
        // Preprocess
        let input_tensor = self.preprocess_frame(frame)?;

        // Convert to f32 tensor
        let mut input_data =
            vec![0f32; self.config.model_input_size * self.config.model_input_size * 3];
        let raw_data = input_tensor.data_bytes()?;

        for (i, &byte) in raw_data.iter().enumerate() {
            if i < input_data.len() {
                input_data[i] = byte as f32;
            }
        }

        // Create input tensor for CoreML
        let input_array = Array4::from_shape_vec(
            (1, self.config.model_input_size, self.config.model_input_size, 3),
            input_data
        )?;

        // Add input to CoreML model
        self.model
            .add_input("input", input_array.into_dyn())
            .map_err(|e| anyhow::anyhow!("Failed to add input to CoreML model: {:?}", e))?;

        // Run inference
        #[cfg(feature = "debug")]
        let start = std::time::Instant::now();

        let mut outputs = self.model
            .predict()
            .map_err(|e| anyhow::anyhow!("CoreML prediction failed: {:?}", e))?;

        #[cfg(feature = "debug")]
        {
            self.frame_counter += 1;
            if self.frame_counter % 30 == 0 {
                let elapsed = start.elapsed();
                println!("Inference time: {:.2}ms", elapsed.as_secs_f32() * 1000.0);
            }
        }

        // Extract Float32 output
        // First, let's see what outputs are available
        #[cfg(feature = "debug")]
        {
            if self.frame_counter == 1 {
                println!("Available outputs: {:?}", outputs.outputs.keys().collect::<Vec<_>>());
            }
        }

        let ml_array = outputs.outputs
            .remove("Identity")
            .ok_or_else(|| anyhow::anyhow!("Output 'Identity' not found"))?;

        // Extract Float16 output and convert to f32
        let output_u16_array = ml_array.extract_to_tensor::<u16>();
        let (output_u16, _) = output_u16_array.into_raw_vec_and_offset();

        // Convert u16 (Float16 bits) to f32
        let output_data: Vec<f32> = output_u16
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect();

        #[cfg(feature = "debug")]
        {
            if self.frame_counter == 1 {
                println!("Output shape: {} values", output_data.len());
                println!("Expected for singlepose: 1 * 1 * 17 * 3 = 51 values");
                if output_data.len() >= 15 {
                    println!("First 5 keypoints (y, x, conf):");
                    for i in 0..5 {
                        let offset = i * 3;
                        println!(
                            "  KP{}: y={:.3}, x={:.3}, conf={:.3}",
                            i,
                            output_data[offset],
                            output_data[offset + 1],
                            output_data[offset + 2]
                        );
                    }
                }
                // Show some stats
                let max_val = output_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min_val = output_data.iter().cloned().fold(f32::INFINITY, f32::min);
                println!("Output range: [{:.3}, {:.3}]", min_val, max_val);
            }
        }

        // Parse singlepose output: [1, 1, 17, 3]
        // Single person with 17 keypoints, each with (y, x, confidence)
        // Note: Output format is [batch, instance, keypoint, coords] = [1, 1, 17, 3]
        let mut all_keypoints = Vec::new();

        // Extract keypoints for the single person
        let mut person_keypoints = Vec::new();

        // Check if we have enough data
        if output_data.len() < self.config.num_keypoints * 3 {
            #[cfg(feature = "debug")]
            println!(
                "Warning: Output size {} is less than expected {}",
                output_data.len(),
                self.config.num_keypoints * 3
            );
            return Ok(Vec::new());
        }

        for i in 0..self.config.num_keypoints {
            let offset = i * 3;
            let kp = [
                output_data[offset + 0], // y
                output_data[offset + 1], // x
                output_data[offset + 2], // confidence
            ];
            person_keypoints.push(kp);
        }

        let high_conf_count = person_keypoints
            .iter()
            .filter(|kp| kp[2] > CONFIDENCE_THRESHOLD)
            .count();

        // Add person if at least 5 keypoints have good confidence (more lenient than average)
        if high_conf_count >= 5 {
            all_keypoints.push(person_keypoints);
        } else {
            #[cfg(feature = "debug")]
            {
                if self.frame_counter <= 3 {
                    println!("Skipping detection: only {} keypoints above threshold", high_conf_count);
                }
            }
        }

        // Apply temporal smoothing
        let smoothed_keypoints = self.apply_temporal_smoothing(all_keypoints);

        // Store for next frame
        self.previous_keypoints = Some(smoothed_keypoints.clone());

        Ok(smoothed_keypoints)
    }

    /// Apply temporal smoothing to reduce jitter
    /// Uses exponential moving average with confidence-based fallback
    /// Handles new people entering/exiting by matching based on spatial proximity
    fn apply_temporal_smoothing(&self, current: MultiPoseKeypoints) -> MultiPoseKeypoints {
        // If no previous frame, return current as-is
        let Some(ref previous) = self.previous_keypoints else {
            return current;
        };

        // If no people detected, return current
        if current.is_empty() {
            return current;
        }

        let mut smoothed = Vec::new();

        for current_person in current.iter() {
            // Find the best matching person from previous frame based on center of mass
            let best_match = self.find_best_match(current_person, previous);

            if let Some(prev_person) = best_match {
                let mut smoothed_person = Vec::new();

                for (kp_idx, current_kp) in current_person.iter().enumerate() {
                    let prev_kp = prev_person.get(kp_idx);

                    if let Some(&prev) = prev_kp {
                        let confidence = current_kp[2];

                        // Determine smoothing factor based on confidence
                        let alpha = if confidence < self.config.low_confidence_threshold {
                            // Low confidence: use more of previous frame
                            self.config.low_confidence_alpha
                        } else {
                            // High confidence: use less of previous frame
                            self.config.smoothing_alpha
                        };

                        // Apply exponential moving average
                        // smoothed = alpha * previous + (1 - alpha) * current
                        let smoothed_kp = [
                            alpha * prev[0] + (1.0 - alpha) * current_kp[0], // y
                            alpha * prev[1] + (1.0 - alpha) * current_kp[1], // x
                            current_kp[2], // keep current confidence
                        ];

                        smoothed_person.push(smoothed_kp);
                    } else {
                        // No previous keypoint, use current
                        smoothed_person.push(*current_kp);
                    }
                }

                smoothed.push(smoothed_person);
            } else {
                // No matching previous person (new person entering frame)
                // Use current detection without smoothing
                smoothed.push(current_person.clone());
            }
        }

        smoothed
    }

    /// Find the best matching person from previous frame based on center of mass
    /// Returns None if no good match is found (person is new to the frame)
    fn find_best_match<'a>(
        &self,
        current_person: &Keypoints,
        previous: &'a MultiPoseKeypoints
    ) -> Option<&'a Keypoints> {
        if previous.is_empty() {
            return None;
        }

        // Calculate center of mass for current person (using high-confidence keypoints)
        let current_center = self.calculate_center_of_mass(current_person);
        if current_center.is_none() {
            // Not enough confident keypoints, don't smooth
            return None;
        }
        let (curr_y, curr_x) = current_center.unwrap();

        // Find closest person in previous frame
        let mut best_match: Option<(usize, f32)> = None;

        for (idx, prev_person) in previous.iter().enumerate() {
            let prev_center = self.calculate_center_of_mass(prev_person);
            if let Some((prev_y, prev_x)) = prev_center {
                // Calculate Euclidean distance
                let distance = ((curr_y - prev_y).powi(2) + (curr_x - prev_x).powi(2)).sqrt();

                // Keep track of closest match
                if let Some((_, best_dist)) = best_match {
                    if distance < best_dist {
                        best_match = Some((idx, distance));
                    }
                } else {
                    best_match = Some((idx, distance));
                }
            }
        }

        // Only return match if distance is reasonable (not too far apart)
        // Threshold of 0.3 means person moved less than 30% of frame size
        if let Some((idx, distance)) = best_match {
            if distance < 0.3 {
                return Some(&previous[idx]);
            }
        }

        None
    }

    /// Calculate center of mass using high-confidence keypoints
    /// Returns (y, x) coordinates normalized to [0, 1]
    fn calculate_center_of_mass(&self, person: &Keypoints) -> Option<(f32, f32)> {
        let mut sum_y = 0.0;
        let mut sum_x = 0.0;
        let mut count = 0;

        for kp in person.iter() {
            let confidence = kp[2];
            // Only use keypoints with reasonable confidence
            if confidence > CONFIDENCE_THRESHOLD {
                sum_y += kp[0];
                sum_x += kp[1];
                count += 1;
            }
        }

        if count > 0 {
            Some((sum_y / (count as f32), sum_x / (count as f32)))
        } else {
            None
        }
    }

    /// Reset temporal smoothing (useful when scene changes drastically)
    pub fn reset_smoothing(&mut self) {
        self.previous_keypoints = None;
        #[cfg(feature = "debug")]
        println!("Temporal smoothing reset");
    }
}
