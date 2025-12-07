use anyhow::Result;
use opencv::{ core, imgproc, prelude::* };
use coreml_rs_fork::{ CoreMLModelOptions, CoreMLModelWithState, ComputePlatform };
use ndarray::Array4;
use half;

#[cfg(feature = "debug")]
use tracing::{ debug, info };

/// COCO keypoints
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

/// (y, x, confidence)
pub type KeypointData = [f32; 3];
pub type Keypoints = Vec<KeypointData>;
pub type MultiPoseKeypoints = Vec<Keypoints>;

/// Pose config
#[derive(Debug, Clone)]
pub struct PoseDetectorConfig {
    pub model_path: String,
    pub input_width: usize,
    pub input_height: usize,
    pub num_keypoints: usize,

    pub mean: [f32; 3],
    pub std: [f32; 3],

    pub simcc_split_ratio: f32,

    /// Low confidence threshold
    pub low_confidence_threshold: f32,
    /// Smoothing alpha
    pub smoothing_alpha: f32,
    /// Low confidence alpha
    pub low_confidence_alpha: f32,
}

impl Default for PoseDetectorConfig {
    fn default() -> Self {
        Self {
            model_path: "models/rtmpose.mlpackage".to_string(),
            input_width: 192,
            input_height: 256,
            num_keypoints: 17,
            mean: [123.675, 116.28, 103.53],
            std: [58.395, 57.12, 57.375],
            simcc_split_ratio: 2.0,
            low_confidence_threshold: 0.2,
            smoothing_alpha: 0.3,
            low_confidence_alpha: 0.7,
        }
    }
}

/// Pose detector
pub struct PoseDetector {
    pub config: PoseDetectorConfig,
    model: CoreMLModelWithState,
    /// Previous keypoints
    previous_keypoints: Option<MultiPoseKeypoints>,
    #[cfg(feature = "debug")]
    pub frame_counter: usize,
}

impl PoseDetector {
    /// New pose detector
    pub fn new(config: PoseDetectorConfig) -> Result<Self> {
        #[cfg(feature = "debug")]
        debug!("Initializing Pose Detector with CoreML...");

        let mut model_options = CoreMLModelOptions::default();
        model_options.compute_platform = ComputePlatform::CpuAndANE;

        let model = CoreMLModelWithState::new(&config.model_path, model_options)
            .load()
            .map_err(|e| anyhow::anyhow!("Failed to load CoreML model: {:?}", e))?;

        #[cfg(feature = "debug")]
        {
            debug!("✓ CoreML model loaded with Neural Engine acceleration!");
            debug!("Model: {}", config.model_path);
            debug!("Input size: {}x{} (WxH)", config.input_width, config.input_height);
        }

        #[cfg(feature = "debug")]
        debug!("✓ Pose Detector ready with temporal smoothing!");

        Ok(Self {
            config,
            model,
            previous_keypoints: None,
            #[cfg(feature = "debug")]
            frame_counter: 0,
        })
    }

    /// Preprocess patch
    fn preprocess_frame(&self, crop_bgr: &Mat) -> Result<Array4<f32>> {
        let mut resized = Mat::default();
        let size = core::Size::new(self.config.input_width as i32, self.config.input_height as i32);
        imgproc::resize(crop_bgr, &mut resized, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;

        let mut rgb = Mat::default();
        imgproc::cvt_color(
            &resized,
            &mut rgb,
            imgproc::COLOR_BGR2RGB,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT
        )?;

        let h = self.config.input_height;
        let w = self.config.input_width;
        let hw = h * w;

        // convert to NCHW
        let raw = rgb.data_bytes()?;
        assert_eq!(raw.len(), h * w * 3);

        let mut nchw = vec![0f32; 3 * h * w];

        let mean = self.config.mean;
        let std = self.config.std;

        for y in 0..h {
            for x in 0..w {
                let src_idx = (y * w + x) * 3;
                let r = raw[src_idx] as f32;
                let g = raw[src_idx + 1] as f32;
                let b = raw[src_idx + 2] as f32;

                let base = y * w + x;
                nchw[0 * hw + base] = (r - mean[0]) / std[0]; // R
                nchw[1 * hw + base] = (g - mean[1]) / std[1]; // G
                nchw[2 * hw + base] = (b - mean[2]) / std[2]; // B
            }
        }

        let input_array = Array4::from_shape_vec((1, 3, h, w), nchw)?;
        Ok(input_array)
    }

    /// Detect poses for a cropped person with CoreML RTMPose model
    pub fn detect_pose(&mut self, person_crop: &Mat) -> Result<MultiPoseKeypoints> {
        // 1. Preprocess crop -> NCHW tensor
        let input_array = self.preprocess_frame(person_crop)?;

        self.model
            .add_input("input", input_array.into_dyn())
            .map_err(|e| anyhow::anyhow!("Failed to add input to CoreML model: {:?}", e))?;

        #[cfg(feature = "debug")]
        let start = std::time::Instant::now();

        let mut outputs = self.model
            .predict()
            .map_err(|e| anyhow::anyhow!("CoreML prediction failed: {:?}", e))?;

        #[cfg(feature = "debug")]
        {
            self.frame_counter += 1;
            if self.frame_counter == 1 {
                debug!("Available outputs: {:?}", outputs.outputs.keys().collect::<Vec<_>>());
            }
            if self.frame_counter % 30 == 0 {
                let elapsed = start.elapsed();
                debug!("RTMPose inference time: {:.2}ms", elapsed.as_secs_f32() * 1000.0);
            }
        }

        // Get SimCC logits
        let x_logits_array = outputs.outputs
            .remove("linear_3")
            .ok_or_else(|| anyhow::anyhow!("Output 'linear_3' not found"))?;
        let y_logits_array = outputs.outputs
            .remove("linear_4")
            .ok_or_else(|| anyhow::anyhow!("Output 'linear_4' not found"))?;

        // Float16 to f32
        let x_u16 = x_logits_array.extract_to_tensor::<u16>();
        let (x_bits, _) = x_u16.into_raw_vec_and_offset();
        let x_logits: Vec<f32> = x_bits
            .iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect();

        let y_u16 = y_logits_array.extract_to_tensor::<u16>();
        let (y_bits, _) = y_u16.into_raw_vec_and_offset();
        let y_logits: Vec<f32> = y_bits
            .iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect();

        let k = self.config.num_keypoints;
        let w_in = self.config.input_width as f32;
        let h_in = self.config.input_height as f32;
        let wx = (w_in * self.config.simcc_split_ratio) as usize;
        let wy = (h_in * self.config.simcc_split_ratio) as usize;

        #[cfg(feature = "debug")]
        {
            if self.frame_counter == 1 {
                debug!("X logits len: {}, expected: {}", x_logits.len(), k * wx);
                debug!("Y logits len: {}, expected: {}", y_logits.len(), k * wy);
            }
        }

        let mut person_keypoints: Keypoints = Vec::with_capacity(k);

        for kp_idx in 0..k {
            let x_start = kp_idx * wx;
            let y_start = kp_idx * wy;

            if x_start + wx > x_logits.len() || y_start + wy > y_logits.len() {
                continue;
            }

            let px_logits = &x_logits[x_start..x_start + wx];
            let py_logits = &y_logits[y_start..y_start + wy];

            // Argmax over SimCC logits (to get pixel coordinates)
            let (mut x_idx, mut max_x) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in px_logits.iter().enumerate() {
                if v > max_x {
                    max_x = v;
                    x_idx = i;
                }
            }

            let (mut y_idx, mut max_y) = (0usize, f32::NEG_INFINITY);
            for (j, &v) in py_logits.iter().enumerate() {
                if v > max_y {
                    max_y = v;
                    y_idx = j;
                }
            }

            let u = (x_idx as f32) / self.config.simcc_split_ratio; // [0,192)
            let v = (y_idx as f32) / self.config.simcc_split_ratio; // [0,256)

            let score = max_x.min(max_y);

            let x_norm = u / w_in;
            let y_norm = v / h_in;

            person_keypoints.push([y_norm, x_norm, score]);
        }

        let mut all = Vec::new();

        let high_conf_count = person_keypoints
            .iter()
            .filter(|kp| kp[2] > CONFIDENCE_THRESHOLD)
            .count();

        if high_conf_count >= 5 {
            all.push(person_keypoints);
        }
        Ok(all)
    }
}
