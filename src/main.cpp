#include <array>
#include <coreml_provider_factory.h>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

const int MODEL_INPUT_SIZE = 256;

// MoveNet outputs 17 keypoints with (y, x, confidence) for each
const int NUM_KEYPOINTS = 17;

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

class MoveNetTracker {
private:
  Ort::Env env;
  Ort::Session session;
  Ort::SessionOptions session_options;
  std::vector<std::string> input_node_name_strings;
  std::vector<std::string> output_node_name_strings;
  std::vector<const char *> input_node_names;
  std::vector<const char *> output_node_names;
  std::vector<int64_t> input_node_dims;

public:
  MoveNetTracker(const std::string &model_path)
      : env(ORT_LOGGING_LEVEL_WARNING, "MoveNetTracker"), session(nullptr) {

    std::cout << "Initializing ONNX Runtime..." << std::endl;

    // Enable CoreML (Apple GPU) acceleration FIRST before other options
    uint32_t coreml_flags = 0;
    coreml_flags |=
        COREML_FLAG_USE_CPU_AND_GPU; // Use both CPU and GPU/Neural Engine
    coreml_flags |= COREML_FLAG_ENABLE_ON_SUBGRAPH; // Enable on subgraphs too

    OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_CoreML(
        session_options, coreml_flags);
    if (status != nullptr) {
      const char *msg = Ort::GetApi().GetErrorMessage(status);
      std::cout << "⚠ CoreML failed: " << msg << std::endl;
      Ort::GetApi().ReleaseStatus(status);
      std::cout << "Falling back to CPU..." << std::endl;
    } else {
      std::cout << "✓ CoreML (GPU/Neural Engine) acceleration enabled!"
                << std::endl;
    }

    // Setup session options for better performance
    session_options.SetIntraOpNumThreads(4); // Use 4 threads for operations
    session_options.SetInterOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

    // Create session
    session = Ort::Session(env, model_path.c_str(), session_options);

    std::cout << "Session created successfully" << std::endl;

    // Get input/output info
    Ort::AllocatorWithDefaultOptions allocator;

    // Input info
    size_t num_input_nodes = session.GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
      auto input_name = session.GetInputNameAllocated(i, allocator);
      input_node_name_strings.push_back(std::string(input_name.get()));

      auto type_info = session.GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      input_node_dims = tensor_info.GetShape();
    }

    // Output info
    size_t num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
      auto output_name = session.GetOutputNameAllocated(i, allocator);
      output_node_name_strings.push_back(std::string(output_name.get()));
    }

    // Convert strings to const char* for use in inference
    for (const auto &name : input_node_name_strings) {
      input_node_names.push_back(name.c_str());
    }
    for (const auto &name : output_node_name_strings) {
      output_node_names.push_back(name.c_str());
    }

    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << "Input shape: [";
    for (size_t i = 0; i < input_node_dims.size(); i++) {
      std::cout << input_node_dims[i];
      if (i < input_node_dims.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }

  cv::Mat preprocessFrame(const cv::Mat &frame) {
    cv::Mat resized, rgb;

    // Resize to model input size
    cv::resize(frame, resized, cv::Size(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE));

    // Convert BGR to RGB
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    return rgb;
  }

  std::vector<std::array<float, 3>> detectPose(const cv::Mat &frame) {
    // Preprocess
    cv::Mat input_tensor = preprocessFrame(frame);

    // Prepare input tensor
    std::vector<int64_t> input_shape = {1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
                                        3};
    size_t input_tensor_size = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3;
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
    auto t_start = std::chrono::high_resolution_clock::now();
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor_obj, 1,
        output_node_names.data(), output_node_names.size());
    auto t_end = std::chrono::high_resolution_clock::now();

    float inference_time =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
            .count() /
        1000.0f;
    static int counter = 0;
    if (++counter % 30 == 0) {
      std::cout << "Inference time: " << inference_time << "ms" << std::endl;
    }

    // Get output
    float *output_data = output_tensors[0].GetTensorMutableData<float>();

    // Parse keypoints
    std::vector<std::array<float, 3>> keypoints;
    for (int i = 0; i < NUM_KEYPOINTS; i++) {
      std::array<float, 3> kp = {
          output_data[i * 3 + 0], // y
          output_data[i * 3 + 1], // x
          output_data[i * 3 + 2]  // confidence
      };
      keypoints.push_back(kp);
    }

    return keypoints;
  }

  void drawKeypoints(cv::Mat &frame,
                     const std::vector<std::array<float, 3>> &keypoints) {
    int height = frame.rows;
    int width = frame.cols;

    // Only draw legs (minimal drawing for maximum FPS)
    const int lineType = cv::LINE_8;

    // Draw leg skeleton only (4 lines total)
    if (keypoints[LEFT_HIP][2] > 0.3 && keypoints[LEFT_KNEE][2] > 0.3) {
      cv::line(frame,
               cv::Point(keypoints[LEFT_HIP][1] * width,
                         keypoints[LEFT_HIP][0] * height),
               cv::Point(keypoints[LEFT_KNEE][1] * width,
                         keypoints[LEFT_KNEE][0] * height),
               cv::Scalar(255, 255, 0), 2, lineType);
    }
    if (keypoints[LEFT_KNEE][2] > 0.3 && keypoints[LEFT_ANKLE][2] > 0.3) {
      cv::line(frame,
               cv::Point(keypoints[LEFT_KNEE][1] * width,
                         keypoints[LEFT_KNEE][0] * height),
               cv::Point(keypoints[LEFT_ANKLE][1] * width,
                         keypoints[LEFT_ANKLE][0] * height),
               cv::Scalar(255, 255, 0), 2, lineType);
    }
    if (keypoints[RIGHT_HIP][2] > 0.3 && keypoints[RIGHT_KNEE][2] > 0.3) {
      cv::line(frame,
               cv::Point(keypoints[RIGHT_HIP][1] * width,
                         keypoints[RIGHT_HIP][0] * height),
               cv::Point(keypoints[RIGHT_KNEE][1] * width,
                         keypoints[RIGHT_KNEE][0] * height),
               cv::Scalar(255, 255, 0), 2, lineType);
    }
    if (keypoints[RIGHT_KNEE][2] > 0.3 && keypoints[RIGHT_ANKLE][2] > 0.3) {
      cv::line(frame,
               cv::Point(keypoints[RIGHT_KNEE][1] * width,
                         keypoints[RIGHT_KNEE][0] * height),
               cv::Point(keypoints[RIGHT_ANKLE][1] * width,
                         keypoints[RIGHT_ANKLE][0] * height),
               cv::Scalar(255, 255, 0), 2, lineType);
    }

    // Draw ankles (2 circles)
    if (keypoints[LEFT_ANKLE][2] > 0.3) {
      cv::circle(frame,
                 cv::Point(keypoints[LEFT_ANKLE][1] * width,
                           keypoints[LEFT_ANKLE][0] * height),
                 6, cv::Scalar(0, 0, 255), -1, lineType);
    }
    if (keypoints[RIGHT_ANKLE][2] > 0.3) {
      cv::circle(frame,
                 cv::Point(keypoints[RIGHT_ANKLE][1] * width,
                           keypoints[RIGHT_ANKLE][0] * height),
                 6, cv::Scalar(0, 0, 255), -1, lineType);
    }
  }
};

int main(int argc, char **argv) {
  std::string model_path = "models/movenet_thunder.onnx";
  int camera_id = 0;

  // Parse arguments: ./footstep_tracker [model_path] [camera_id]
  if (argc > 1) {
    model_path = argv[1];
  }
  if (argc > 2) {
    camera_id = std::atoi(argv[2]);
  }

  try {
    // Initialize MoveNet tracker
    MoveNetTracker tracker(model_path);

    // Open webcam with AVFoundation backend (better for macOS)
    cv::VideoCapture cap(camera_id, cv::CAP_AVFOUNDATION);
    if (!cap.isOpened()) {
      std::cerr << "Error: Could not open camera " << camera_id << std::endl;
      std::cerr << "Try a different camera ID with: ./footstep_tracker "
                   "models/movenet_lightning.onnx <camera_id>"
                << std::endl;
      return -1;
    }

    std::cout << "Press 'q' to quit" << std::endl;

    cv::Mat frame;
    while (true) {
      cap >> frame;
      if (frame.empty())
        break;

      auto keypoints = tracker.detectPose(frame);
      tracker.drawKeypoints(frame, keypoints);

      cv::imshow("Footstep Tracker", frame);
      if (cv::waitKey(1) == 'q')
        break;
    }

    cap.release();
    cv::destroyAllWindows();

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
