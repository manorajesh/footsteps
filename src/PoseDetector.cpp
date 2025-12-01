#include "../include/PoseDetector.hpp"
#include <coreml_provider_factory.h>

// Compile-time logging macros
#ifdef POSE_DETECTOR_DEBUG
#include <chrono>
#include <iostream>
#define LOG_INFO(msg) std::cout << msg << std::endl
#else
#define LOG_INFO(msg) ((void)0)
#endif

PoseDetector::PoseDetector(const PoseDetectorConfig &config)
    : config_(config), env_(ORT_LOGGING_LEVEL_WARNING, "PoseDetector"),
      session_(nullptr) {

  LOG_INFO("Initializing Pose Detector...");

  initializeSession();

  LOG_INFO("✓ Pose Detector ready!");
}

void PoseDetector::initializeSession() {
  // Configure CoreML acceleration if requested
  if (config_.use_coreml) {
    // Set CoreML session options for optimal performance
    session_options_.AddConfigEntry("session.coreml.ModelFormat", "MLProgram");
    session_options_.AddConfigEntry("session.coreml.MLComputeUnits", "ALL");
    session_options_.AddConfigEntry("session.coreml.RequireStaticInputShapes",
                                    "1");
    session_options_.AddConfigEntry("session.coreml.EnableOnSubgraphs", "1");
    session_options_.AddConfigEntry("session.coreml.SpecializationStrategy",
                                    "FastPrediction");
    session_options_.AddConfigEntry(
        "session.coreml.AllowLowPrecisionAccumulationOnGPU", "1");

    uint32_t coreml_flags = 0;
    if (config_.use_cpu_and_gpu) {
      coreml_flags |= COREML_FLAG_USE_CPU_AND_GPU;
    }
    coreml_flags |= COREML_FLAG_ENABLE_ON_SUBGRAPH;

    OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_CoreML(
        session_options_, coreml_flags);

    if (status != nullptr) {
#ifdef POSE_DETECTOR_DEBUG
      const char *msg = Ort::GetApi().GetErrorMessage(status);
      std::cout << "⚠ CoreML failed: " << msg << std::endl;
      std::cout << "Falling back to CPU..." << std::endl;
#endif
      Ort::GetApi().ReleaseStatus(status);
    } else {
      LOG_INFO("✓ CoreML (GPU/Neural Engine) acceleration enabled!");
    }
  }

  // Configure performance settings
  session_options_.SetIntraOpNumThreads(config_.intra_op_threads);
  session_options_.SetInterOpNumThreads(config_.inter_op_threads);
  session_options_.SetGraphOptimizationLevel(config_.optimization_level);

  if (config_.enable_parallel_execution) {
    session_options_.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
  }

  // Create session
  session_ = Ort::Session(env_, config_.model_path.c_str(), session_options_);

  // Get input/output info
  Ort::AllocatorWithDefaultOptions allocator;

  // Input info
  size_t num_input_nodes = session_.GetInputCount();
  for (size_t i = 0; i < num_input_nodes; i++) {
    auto input_name = session_.GetInputNameAllocated(i, allocator);
    input_node_name_strings_.push_back(std::string(input_name.get()));

    auto type_info = session_.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    input_node_dims_ = tensor_info.GetShape();
  }

  // Output info
  size_t num_output_nodes = session_.GetOutputCount();
  for (size_t i = 0; i < num_output_nodes; i++) {
    auto output_name = session_.GetOutputNameAllocated(i, allocator);
    output_node_name_strings_.push_back(std::string(output_name.get()));
  }

  // Convert strings to const char* for use in inference
  for (const auto &name : input_node_name_strings_) {
    input_node_names_.push_back(name.c_str());
  }
  for (const auto &name : output_node_name_strings_) {
    output_node_names_.push_back(name.c_str());
  }

#ifdef POSE_DETECTOR_DEBUG
  std::cout << "Model: " << config_.model_path << std::endl;
  std::cout << "Input shape: [";
  for (size_t i = 0; i < input_node_dims_.size(); i++) {
    std::cout << input_node_dims_[i];
    if (i < input_node_dims_.size() - 1)
      std::cout << ", ";
  }
  std::cout << "]" << std::endl;
#endif
}

cv::Mat PoseDetector::preprocessFrame(const cv::Mat &frame) {
  cv::Mat resized, rgb;

  // Resize to model input size
  cv::resize(frame, resized,
             cv::Size(config_.model_input_size, config_.model_input_size));

  // Convert BGR to RGB
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  return rgb;
}

MultiPoseKeypoints PoseDetector::detectPose(const cv::Mat &frame) {
  // Preprocess
  cv::Mat input_tensor = preprocessFrame(frame);

  // Prepare input tensor
  std::vector<int64_t> input_shape = {1, config_.model_input_size,
                                      config_.model_input_size, 3};
  size_t input_tensor_size =
      config_.model_input_size * config_.model_input_size * 3;
  std::vector<int32_t> input_tensor_values(input_tensor_size);

  // Convert uint8 to int32
  const uint8_t *src = input_tensor.data;
  int32_t *dst = input_tensor_values.data();
  for (size_t i = 0; i < input_tensor_size; i++) {
    dst[i] = static_cast<int32_t>(src[i]);
  }

  // Create input tensor
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor_obj = Ort::Value::CreateTensor<int32_t>(
      memory_info, input_tensor_values.data(), input_tensor_size,
      input_shape.data(), input_shape.size());

  // Run inference
#ifdef POSE_DETECTOR_DEBUG
  auto t_start = std::chrono::high_resolution_clock::now();
#endif
  auto output_tensors = session_.Run(
      Ort::RunOptions{nullptr}, input_node_names_.data(), &input_tensor_obj, 1,
      output_node_names_.data(), output_node_names_.size());
#ifdef POSE_DETECTOR_DEBUG
  auto t_end = std::chrono::high_resolution_clock::now();

  // Log inference time periodically
  if (++frame_counter_ % config_.inference_log_frequency == 0) {
    float inference_time =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
            .count() /
        1000.0f;
    std::cout << "Inference time: " << inference_time << "ms" << std::endl;
  }
#endif

  // Get output
  float *output_data = output_tensors[0].GetTensorMutableData<float>();

  // Parse multipose output: [1, 6, 56]
  // Each person has 56 values: 17 keypoints * 3 + 4 bbox + 1 score
  all_keypoints_.clear();

  for (int person = 0; person < config_.max_people; person++) {
    int person_offset = person * 56;

    // Check overall detection score (last value for this person)
    float detection_score = output_data[person_offset + 55];

    // Skip people with low detection confidence
    if (detection_score < 0.3f) {
      continue;
    }

    Keypoints person_keypoints;
    for (int i = 0; i < config_.num_keypoints; i++) {
      KeypointData kp = {
          output_data[person_offset + i * 3 + 0], // y
          output_data[person_offset + i * 3 + 1], // x
          output_data[person_offset + i * 3 + 2]  // confidence
      };
      person_keypoints.push_back(kp);
    }
    all_keypoints_.push_back(person_keypoints);
  }

  return all_keypoints_;
}
