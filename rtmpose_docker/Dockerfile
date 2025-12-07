FROM --platform=linux/amd64 python:3.10-slim

# System deps for OpenCV/mmcv etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN python -m pip install --upgrade pip

# 🔒 PIN NUMPY < 2 FIRST
RUN python -m pip install "numpy==1.26.4"

# Torch (CPU is enough for conversion)
RUN python -m pip install \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

# OpenMMLab stack (matching versions)
RUN python -m pip install -U openmim
RUN python -m pip install chumpy --no-build-isolation
RUN mim install "mmengine>=0.7.0"
RUN mim install "mmcv==2.1.0"
RUN mim install "mmdet==3.2.0"
RUN mim install "mmpose==1.3.2"

# CoreMLTools
RUN python -m pip install coremltools==7.1

CMD ["/bin/bash"]