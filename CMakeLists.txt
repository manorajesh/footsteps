cmake_minimum_required(VERSION 3.15)
project(footstep_tracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build in Release mode by default for better performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Enable optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find ONNX Runtime
# First try the local CoreML-enabled version in dependencies folder
file(GLOB ONNXRUNTIME_DIRS "${CMAKE_SOURCE_DIR}/dependencies/onnxruntime-osx-arm64-*")
if(ONNXRUNTIME_DIRS)
    list(GET ONNXRUNTIME_DIRS 0 ONNXRUNTIME_ROOT)
    message(STATUS "Using local ONNX Runtime with CoreML: ${ONNXRUNTIME_ROOT}")
elseif(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
endif()

if(ONNXRUNTIME_ROOT)
    include_directories(${ONNXRUNTIME_ROOT}/include)
    link_directories(${ONNXRUNTIME_ROOT}/lib)
else()
    # Fallback to Homebrew on macOS
    if(APPLE)
        execute_process(
            COMMAND brew --prefix onnxruntime
            OUTPUT_VARIABLE ONNXRUNTIME_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(ONNXRUNTIME_PREFIX)
            include_directories(${ONNXRUNTIME_PREFIX}/include/onnxruntime)
            link_directories(${ONNXRUNTIME_PREFIX}/lib)
        endif()
    endif()
endif()

# Add executable
add_executable(footstep_tracker src/main.cpp)

# Link libraries
target_link_libraries(footstep_tracker 
    ${OpenCV_LIBS}
    onnxruntime
)

# Set rpath for dynamic library loading
if(ONNXRUNTIME_ROOT)
    set_target_properties(footstep_tracker PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
    )
endif()
