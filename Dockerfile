FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG GAUSSIAN_SPLATTING_REF=54c035f7834b564019656c3e3fcc3646292f727d

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

RUN git init ${GAUSSIAN_SPLATTING_ROOT} \
    && cd ${GAUSSIAN_SPLATTING_ROOT} \
    && git remote add origin https://github.com/graphdeco-inria/gaussian-splatting.git \
    && git fetch --depth 1 origin ${GAUSSIAN_SPLATTING_REF} \
    && git checkout FETCH_HEAD \
    && git submodule update --init --recursive --depth 1

RUN python3 -m pip install \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN python3 -m pip install \
    numpy==1.26.4 \
    pillow==10.4.0 \
    plyfile==1.1.2 \
    tqdm==4.67.1 \
    opencv-python-headless==4.10.0.84 \
    joblib==1.4.2

RUN python3 -m pip install --no-build-isolation \
    ${GAUSSIAN_SPLATTING_ROOT}/submodules/diff-gaussian-rasterization \
    ${GAUSSIAN_SPLATTING_ROOT}/submodules/simple-knn \
    ${GAUSSIAN_SPLATTING_ROOT}/submodules/fused-ssim

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

COPY . .
RUN python3 scripts/smoke_check.py

CMD ["python3", "handler.py"]
