#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct Keypoint {
    float x;
    float y;
    float conf;
};

int main(int argc, char** argv) {
    // --------- Config ---------
    const std::string modelPath = "pose.onnx";
    const int inputWidth  = 256;  // set to your model's expected width
    const int inputHeight = 256;  // set to your model's expected height
    const float confThreshold = 0.5f;

    // Camera index (0 = default cam; change if needed)
    int cameraIndex = 0;

    // --------- OpenCV camera ---------
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "Error: could not open camera " << cameraIndex << std::endl;
        return 1;
    }

    // Optional: set resolution (depends on camera support)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 60);

    // --------- ONNX Runtime setup ---------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pose_app");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // If you built ONNX Runtime with CUDA and want GPU, uncomment this:
    // #include <cuda_provider_factory.h>  // You may need this header
    // OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

    Ort::Session session(env, modelPath.c_str(), sessionOptions);

    // Get input / output names
    Ort::AllocatorWithDefaultOptions allocator;

    char* inputNamePtr = session.GetInputName(0, allocator);
    std::string inputName(inputNamePtr);
    allocator.Free(inputNamePtr);

    char* outputNamePtr = session.GetOutputName(0, allocator);
    std::string outputName(outputNamePtr);
    allocator.Free(outputNamePtr);

    std::cout << "Input name: " << inputName << std::endl;
    std::cout << "Output name: " << outputName << std::endl;

    // Input shape (we’ll override height/width with our config)
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

    if (inputDims.size() != 4) {
        std::cerr << "Unexpected input dims; expected 4D [N,C,H,W]" << std::endl;
        return 1;
    }

    inputDims[0] = 1; // batch
    int64_t channels = inputDims[1];
    inputDims[2] = inputHeight;
    inputDims[3] = inputWidth;

    if (channels != 3) {
        std::cerr << "Expected 3 channels (RGB); got " << channels << std::endl;
        return 1;
    }

    size_t inputTensorSize = 1 * channels * inputHeight * inputWidth;

    // Names for Run()
    const char* inputNames[] = { inputName.c_str() };
    const char* outputNames[] = { outputName.c_str() };

    std::vector<float> inputTensorValues(inputTensorSize);

    // --------- Main loop ---------
    cv::Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Failed to read frame from camera" << std::endl;
            break;
        }

        cv::Mat frameRGB;
        cv::cvtColor(frame, frameRGB, cv::COLOR_BGR2RGB);

        cv::Mat resized;
        cv::resize(frameRGB, resized, cv::Size(inputWidth, inputHeight));

        // Convert to float32 [0,1], NCHW
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

        // HWC -> CHW
        std::vector<cv::Mat> chw(3);
        for (int i = 0; i < 3; ++i) {
            chw[i] = cv::Mat(inputHeight, inputWidth, CV_32FC1, inputTensorValues.data() + i * inputHeight * inputWidth);
        }
        cv::split(resized, chw);

        // Create tensor
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo,
            inputTensorValues.data(),
            inputTensorSize,
            inputDims.data(),
            inputDims.size()
        );

        // Run inference
        auto outputTensors = session.Run(
            Ort::RunOptions{ nullptr },
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1
        );

        if (outputTensors.size() != 1 || !outputTensors[0].IsTensor()) {
            std::cerr << "Unexpected output from model" << std::endl;
            continue;
        }

        // Parse output: assume [1, N, 3] -> (x, y, conf), normalized coords
        Ort::Value& outputTensor = outputTensors[0];
        auto outputTypeInfo = outputTensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> outputDims = outputTypeInfo.GetShape();

        if (outputDims.size() != 3 || outputDims[0] != 1 || outputDims[2] != 3) {
            std::cerr << "Unexpected output dimensions, expected [1, N, 3]" << std::endl;
            continue;
        }

        int64_t numKeypoints = outputDims[1];
        size_t outputTensorSize = 1 * numKeypoints * 3;

        float* outputData = outputTensor.GetTensorMutableData<float>();

        std::vector<Keypoint> keypoints;
        keypoints.reserve(numKeypoints);
        for (int i = 0; i < numKeypoints; ++i) {
            float x = outputData[i * 3 + 0];
            float y = outputData[i * 3 + 1];
            float c = outputData[i * 3 + 2];
            keypoints.push_back({ x, y, c });
        }

        // Draw keypoints on the original frame
        for (const auto& kp : keypoints) {
            if (kp.conf < confThreshold) continue;

            // Assuming x,y are normalized [0,1]
            int px = static_cast<int>(kp.x * frame.cols);
            int py = static_cast<int>(kp.y * frame.rows);

            cv::circle(frame, cv::Point(px, py), 4, cv::Scalar(0, 255, 0), -1);
        }

        cv::imshow("Pose", frame);
        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27 || key == 'q') { // ESC or 'q'
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}