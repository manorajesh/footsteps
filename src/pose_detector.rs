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
}

impl Default for PoseDetectorConfig {
    fn default() -> Self {
        Self {
            model_path: "models/movenet_multipose.mlmodelc".to_string(),
            model_input_size: 256,
            num_keypoints: 17,
            max_people: 6,
        }
    }
}

/// Pose detector using CoreML
pub struct PoseDetector {
    config: PoseDetectorConfig,
    model: CoreMLModelWithState,
    #[cfg(feature = "debug")]
    frame_counter: usize,
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
        println!("✓ Pose Detector ready!");

        Ok(Self {
            config,
            model,
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

        // Extract Float16 output and convert to f32
        let ml_array = outputs.outputs
            .remove("Identity")
            .ok_or_else(|| anyhow::anyhow!("Output 'Identity' not found"))?;

        let output_u16_array = ml_array.extract_to_tensor::<u16>();
        let (output_u16, _) = output_u16_array.into_raw_vec_and_offset();

        // Convert u16 (Float16 bits) to f32
        let output_data: Vec<f32> = output_u16
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect();

        // Parse multipose output: [1, 6, 56]
        // Each person has 56 values: 17 keypoints * 3 + 4 bbox + 1 score
        let mut all_keypoints = Vec::new();

        for person in 0..self.config.max_people {
            let person_offset = person * 56;

            // Check overall detection score (last value for this person)
            let detection_score = output_data[person_offset + 55];

            // Skip people with low detection confidence
            if detection_score < 0.3 {
                continue;
            }

            let mut person_keypoints = Vec::new();
            for i in 0..self.config.num_keypoints {
                let kp = [
                    output_data[person_offset + i * 3 + 0], // y
                    output_data[person_offset + i * 3 + 1], // x
                    output_data[person_offset + i * 3 + 2], // confidence
                ];
                person_keypoints.push(kp);
            }
            all_keypoints.push(person_keypoints);
        }

        Ok(all_keypoints)
    }
}
