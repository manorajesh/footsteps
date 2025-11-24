#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#include "PoseDetector.hpp"
#include <opencv2/opencv.hpp>

// Helper functions for visualizing pose detection results

inline void drawAnkle(cv::Mat &frame, const Keypoints &keypoints,
                      float confidence_threshold = 0.3f) {
  int height = frame.rows;
  int width = frame.cols;

  const int lineType = cv::LINE_8;

  // Draw ankles
  if (keypoints[LEFT_ANKLE][2] > confidence_threshold) {
    cv::circle(frame,
               cv::Point(keypoints[LEFT_ANKLE][1] * width,
                         keypoints[LEFT_ANKLE][0] * height),
               20, cv::Scalar(0, 0, 255), -1, lineType);
  }
  if (keypoints[RIGHT_ANKLE][2] > confidence_threshold) {
    cv::circle(frame,
               cv::Point(keypoints[RIGHT_ANKLE][1] * width,
                         keypoints[RIGHT_ANKLE][0] * height),
               20, cv::Scalar(0, 0, 255), -1, lineType);
  }
}

inline void drawAllKeypoints(cv::Mat &frame, const Keypoints &keypoints,
                             float confidence_threshold = 0.3f) {
  int height = frame.rows;
  int width = frame.cols;

  // Draw all keypoints
  for (size_t i = 0; i < keypoints.size(); i++) {
    if (keypoints[i][2] > confidence_threshold) {
      cv::circle(frame,
                 cv::Point(keypoints[i][1] * width, keypoints[i][0] * height),
                 4, cv::Scalar(0, 255, 0), -1);
    }
  }

  // Define skeleton connections
  std::vector<std::pair<int, int>> connections = {
      {NOSE, LEFT_EYE},
      {NOSE, RIGHT_EYE},
      {LEFT_EYE, LEFT_EAR},
      {RIGHT_EYE, RIGHT_EAR},
      {NOSE, LEFT_SHOULDER},
      {NOSE, RIGHT_SHOULDER},
      {LEFT_SHOULDER, RIGHT_SHOULDER},
      {LEFT_SHOULDER, LEFT_ELBOW},
      {LEFT_ELBOW, LEFT_WRIST},
      {RIGHT_SHOULDER, RIGHT_ELBOW},
      {RIGHT_ELBOW, RIGHT_WRIST},
      {LEFT_SHOULDER, LEFT_HIP},
      {RIGHT_SHOULDER, RIGHT_HIP},
      {LEFT_HIP, RIGHT_HIP},
      {LEFT_HIP, LEFT_KNEE},
      {LEFT_KNEE, LEFT_ANKLE},
      {RIGHT_HIP, RIGHT_KNEE},
      {RIGHT_KNEE, RIGHT_ANKLE}};

  // Draw skeleton
  for (const auto &[start, end] : connections) {
    if (keypoints[start][2] > confidence_threshold &&
        keypoints[end][2] > confidence_threshold) {
      cv::line(
          frame,
          cv::Point(keypoints[start][1] * width, keypoints[start][0] * height),
          cv::Point(keypoints[end][1] * width, keypoints[end][0] * height),
          cv::Scalar(255, 0, 0), 2);
    }
  }
}

#endif // VISUALIZATION_HPP
