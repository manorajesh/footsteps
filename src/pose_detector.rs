use anyhow::{ Context, Result };
use opencv::{ core, imgproc, prelude::* };
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::{ Session, builder::GraphOptimizationLevel };
use ort::value::Value;

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
    pub use_coreml: bool,
    pub intra_op_threads: usize,
    pub inter_op_threads: usize,
}

impl Default for PoseDetectorConfig {
    fn default() -> Self {
        Self {
            model_path: "models/movenet_multipose.onnx".to_string(),
            model_input_size: 256,
            num_keypoints: 17,
            max_people: 6,
            use_coreml: true,
            intra_op_threads: 4,
            inter_op_threads: 4,
        }
    }
}

/// Pose detector using ONNX Runtime
pub struct PoseDetector {
    config: PoseDetectorConfig,
    session: Session,
    #[cfg(feature = "debug")]
    frame_counter: usize,
}

impl PoseDetector {
    /// Create a new pose detector with the given configuration
    pub fn new(config: PoseDetectorConfig) -> Result<Self> {
        #[cfg(feature = "debug")]
        println!("Initializing Pose Detector...");

        // Configure CoreML acceleration if requested
        let session = if config.use_coreml {
            #[cfg(feature = "debug")]
            {
                println!("Attempting CoreML initialization...");
                match
                    Session::builder()?
                        .with_execution_providers([CoreMLExecutionProvider::default().build()])?
                        .with_intra_threads(config.intra_op_threads)?
                        .with_inter_threads(config.inter_op_threads)?
                        .with_optimization_level(GraphOptimizationLevel::Level3)?
                        .commit_from_file(&config.model_path)
                {
                    Ok(session) => {
                        println!("✓ CoreML (GPU/Neural Engine) acceleration enabled!");
                        println!("Note: CoreML may still fall back to CPU for unsupported ops");
                        session
                    }
                    Err(e) => {
                        println!("⚠ CoreML initialization failed: {}", e);
                        println!("Falling back to CPU-only execution...");
                        Session::builder()?
                            .with_intra_threads(config.intra_op_threads)?
                            .with_inter_threads(config.inter_op_threads)?
                            .with_optimization_level(GraphOptimizationLevel::Level3)?
                            .commit_from_file(&config.model_path)
                            .context("Failed to load ONNX model")?
                    }
                }
            }

            #[cfg(not(feature = "debug"))]
            Session::builder()?
                .with_execution_providers([CoreMLExecutionProvider::default().build()])?
                .with_intra_threads(config.intra_op_threads)?
                .with_inter_threads(config.inter_op_threads)?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_file(&config.model_path)
                .context("Failed to load ONNX model")?
        } else {
            Session::builder()?
                .with_intra_threads(config.intra_op_threads)?
                .with_inter_threads(config.inter_op_threads)?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_file(&config.model_path)
                .context("Failed to load ONNX model")?
        };

        #[cfg(feature = "debug")]
        {
            println!("Model: {}", config.model_path);
            println!("Input size: {}", config.model_input_size);
        }

        #[cfg(feature = "debug")]
        println!("✓ Pose Detector ready!");

        Ok(Self {
            config,
            session,
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

        // Convert to i32 tensor
        let mut input_data =
            vec![0i32; self.config.model_input_size * self.config.model_input_size * 3];
        let raw_data = input_tensor.data_bytes()?;

        for (i, &byte) in raw_data.iter().enumerate() {
            if i < input_data.len() {
                input_data[i] = byte as i32;
            }
        }

        // Create input tensor as ort::Value
        let input_array = ndarray::Array4::from_shape_vec(
            (1, self.config.model_input_size, self.config.model_input_size, 3),
            input_data
        )?;
        let input = Value::from_array(input_array)?;

        // Run inference
        #[cfg(feature = "debug")]
        let start = std::time::Instant::now();

        let outputs = self.session.run(ort::inputs![input])?;

        #[cfg(feature = "debug")]
        {
            self.frame_counter += 1;
            if self.frame_counter % 30 == 0 {
                let elapsed = start.elapsed();
                println!("Inference time: {:.2}ms", elapsed.as_secs_f32() * 1000.0);
            }
        }

        // Parse output
        let output = &outputs[0];
        let tensor = output.try_extract_tensor::<f32>()?;
        let output_data = tensor.1; // (shape, data) tuple

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

    pub fn config(&self) -> &PoseDetectorConfig {
        &self.config
    }
}
