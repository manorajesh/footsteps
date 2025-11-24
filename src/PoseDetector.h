#ifndef POSE_DETECTOR_H
#define POSE_DETECTOR_H

#include <array>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Keypoint indices
enum Keypoint {
  NOSE = 0,
  LEFT_EYE = 1,
  RIGHT_EYE = 2,
  LEFT_EAR = 3,
  RIGHT_EAR = 4,
  LEFT_SHOULDER = 5,
  RIGHT_SHOULDER = 6,
  LEFT_ELBOW = 7,
  RIGHT_ELBOW = 8,
  LEFT_WRIST = 9,
  RIGHT_WRIST = 10,
  LEFT_HIP = 11,
  RIGHT_HIP = 12,
  LEFT_KNEE = 13,
  RIGHT_KNEE = 14,
  LEFT_ANKLE = 15,
  RIGHT_ANKLE = 16
};

// Each keypoint contains (y, x, confidence)
using KeypointData = std::array<float, 3>;
using Keypoints = std::vector<KeypointData>;

struct PoseDetectorConfig {
  std::string model_path = "models/movenet_thunder.onnx";
  int model_input_size = 256;
  int num_keypoints = 17;

  // Hardware acceleration
  bool use_coreml = true;
  bool use_cpu_and_gpu = true;

  // Performance settings
  int intra_op_threads = 4;
  int inter_op_threads = 4;
  bool enable_parallel_execution = true;
  GraphOptimizationLevel optimization_level =
      GraphOptimizationLevel::ORT_ENABLE_ALL;

  // Debug settings (only used when compiled with POSE_DETECTOR_DEBUG)
  int inference_log_frequency = 30; // Log every N frames
};

class PoseDetector {
public:
  explicit PoseDetector(
      const PoseDetectorConfig &config = PoseDetectorConfig());
  ~PoseDetector() = default;

  // Prevent copying
  PoseDetector(const PoseDetector &) = delete;
  PoseDetector &operator=(const PoseDetector &) = delete;

  // Allow moving
  PoseDetector(PoseDetector &&) = default;
  PoseDetector &operator=(PoseDetector &&) = default;

  // Main detection method - returns keypoints for the frame
  Keypoints detectPose(const cv::Mat &frame);

  // Get the last detected keypoints without running inference again
  const Keypoints &getKeypoints() const { return last_keypoints_; }

  // Configuration access
  const PoseDetectorConfig &getConfig() const { return config_; }

private:
  cv::Mat preprocessFrame(const cv::Mat &frame);
  void initializeSession();

  PoseDetectorConfig config_;
  Ort::Env env_;
  Ort::Session session_;
  Ort::SessionOptions session_options_;

  std::vector<std::string> input_node_name_strings_;
  std::vector<std::string> output_node_name_strings_;
  std::vector<const char *> input_node_names_;
  std::vector<const char *> output_node_names_;
  std::vector<int64_t> input_node_dims_;

  Keypoints last_keypoints_;
  int frame_counter_ = 0;
};

#endif // POSE_DETECTOR_H
