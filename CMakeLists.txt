cmake_minimum_required(VERSION 3.14)
project(PoseTracker CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Adjust these paths to where OpenCV and ONNX Runtime are installed
set(OpenCV_DIR "/path/to/opencv/build")
set(ONNXRUNTIME_ROOT "/path/to/onnxruntime")

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
)

link_directories(
    ${ONNXRUNTIME_ROOT}/lib
)

add_executable(pose_tracker main.cpp)

target_link_libraries(pose_tracker
    ${OpenCV_LIBS}
    onnxruntime
)