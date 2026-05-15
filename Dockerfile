FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV GAUSSIAN_SPLATTING_ROOT=/opt/gaussian-splatting
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV QT_QPA_PLATFORM=offscreen
ENV XDG_RUNTIME_DIR=/tmp/runtime-root

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    python3-dev \
    python3-pip \
    git \
    curl \
    ffmpeg \
    colmap \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @playcanvas/splat-transform@2.1.1 \
    && npm cache clean --force \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN git clone --depth 1 --recursive --shallow-submodules https://github.com/graphdeco-inria/gaussian-splatting.git ${GAUSSIAN_SPLATTING_ROOT}

RUN python3 -m pip install \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN python3 -m pip install \
    plyfile \
    tqdm \
    opencv-python-headless

RUN python3 -c "import cv2"

RUN python3 -m pip install --no-build-isolation \
    ${GAUSSIAN_SPLATTING_ROOT}/submodules/diff-gaussian-rasterization \
    ${GAUSSIAN_SPLATTING_ROOT}/submodules/simple-knn \
    ${GAUSSIAN_SPLATTING_ROOT}/submodules/fused-ssim

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

COPY . .

CMD ["python3", "handler.py"]
