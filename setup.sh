#!/bin/bash

# Footstep Tracker Setup Script
# This script installs all dependencies and downloads the ONNX Runtime with CoreML support

set -e  # Exit on error

echo "🚀 Setting up Footstep Tracker..."
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS. For other platforms, please install dependencies manually."
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew is not installed. Please install it from https://brew.sh/"
    exit 1
fi

echo "✓ Homebrew found"
echo ""

# Install OpenCV
echo "📦 Installing OpenCV..."
if brew list opencv &>/dev/null; then
    echo "✓ OpenCV already installed"
else
    brew install opencv
    echo "✓ OpenCV installed"
fi
echo ""

# Create dependencies directory if it doesn't exist
if [ ! -d "dependencies" ]; then
    echo "📁 Creating dependencies directory..."
    mkdir dependencies
    echo "✓ Dependencies directory created"
fi
echo ""

# Check for ONNX Runtime with CoreML
ONNX_VERSION="1.23.2"
ONNX_DIR="dependencies/onnxruntime-osx-arm64-${ONNX_VERSION}"
ONNX_ARCHIVE="onnxruntime-osx-arm64-${ONNX_VERSION}.tgz"
ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_ARCHIVE}"

if [ -d "$ONNX_DIR" ]; then
    echo "✓ ONNX Runtime ${ONNX_VERSION} with CoreML already exists"
else
    echo "📦 Downloading ONNX Runtime ${ONNX_VERSION} with CoreML support..."
    
    # Check architecture
    if [[ $(uname -m) == "arm64" ]]; then
        echo "✓ Detected Apple Silicon (ARM64)"
        curl -L -o "$ONNX_ARCHIVE" "$ONNX_URL"
        
        echo "📦 Extracting ONNX Runtime..."
        tar -xzf "$ONNX_ARCHIVE" -C dependencies/
        rm "$ONNX_ARCHIVE"
        
        echo "✓ ONNX Runtime ${ONNX_VERSION} with CoreML installed"
    else
        echo "❌ Intel Macs are not supported by this setup script."
        echo "Please download the appropriate ONNX Runtime version from:"
        echo "https://github.com/microsoft/onnxruntime/releases"
        exit 1
    fi
fi
echo ""

# Create models directory if it doesn't exist
if [ ! -d "models" ]; then
    echo "📁 Creating models directory..."
    mkdir models
    echo "✓ Models directory created"
else
    echo "✓ Models directory exists"
fi
echo ""

# Download MoveNet models
THUNDER_MODEL="models/movenet_thunder.onnx"
LIGHTNING_MODEL="models/movenet_lightning.onnx"

echo "📦 Downloading MoveNet models..."

# Download Thunder model
if [ -f "$THUNDER_MODEL" ]; then
    echo "✓ MoveNet Thunder already exists"
else
    echo "Downloading MoveNet Thunder (more accurate, slower)..."
    curl -L -o "$THUNDER_MODEL" "https://huggingface.co/Xenova/movenet-singlepose-thunder/resolve/main/onnx/model.onnx"
    if [ -f "$THUNDER_MODEL" ]; then
        echo "✓ MoveNet Thunder downloaded"
    else
        echo "⚠️  Failed to download MoveNet Thunder"
    fi
fi

# Download Lightning model
if [ -f "$LIGHTNING_MODEL" ]; then
    echo "✓ MoveNet Lightning already exists"
else
    echo "Downloading MoveNet Lightning (faster, less accurate)..."
    curl -L -o "$LIGHTNING_MODEL" "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx"
    if [ -f "$LIGHTNING_MODEL" ]; then
        echo "✓ MoveNet Lightning downloaded"
    else
        echo "⚠️  Failed to download MoveNet Lightning"
    fi
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Build the project:"
echo "   mkdir -p build && cd build"
echo "   cmake .."
echo "   cmake --build . && cd .."
echo "3. Run the tracker:"
echo "   ./build/footstep_tracker"
echo ""
