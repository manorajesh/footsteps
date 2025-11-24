#include "PoseDetector.h"
#include <iostream>
#include <opencv2/opencv.hpp>

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
    // Configure and initialize the pose detector
    PoseDetectorConfig config;
    config.model_path = model_path;
    config.use_coreml = true;

    PoseDetector detector(config);

    // Open webcam with AVFoundation backend (better for macOS)
    cv::VideoCapture cap(camera_id, cv::CAP_AVFOUNDATION);
    if (!cap.isOpened()) {
      std::cerr << "Error: Could not open camera " << camera_id << std::endl;
      std::cerr << "Try a different camera ID with: ./footstep_tracker "
                   "models/movenet_lightning.onnx <camera_id>"
                << std::endl;
      return -1;
    }

    std::cout << "Press Ctrl+C to quit" << std::endl;

    cv::Mat frame;
    while (true) {
      cap >> frame;
      if (frame.empty())
        break;

      // Detect pose and get keypoints
      Keypoints keypoints = detector.detectPose(frame);

      // Keypoints are now exposed and ready to use
      // Each keypoint: keypoints[i] = {y, x, confidence}
      // Example: Access left ankle
      // float ankle_y = keypoints[LEFT_ANKLE][0];
      // float ankle_x = keypoints[LEFT_ANKLE][1];
      // float confidence = keypoints[LEFT_ANKLE][2];
    }

    cap.release();

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
