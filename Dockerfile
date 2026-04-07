# PyTorch 2.5.1 for SAM2 (requires torch>=2.5.1). For older stack only use 2.3.1 base.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Build as root to install deps, then switch to non-root user
ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# System dependencies (OpenCV runtime, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-opencv \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# SAM2 (Segment Anything Model 2) for pseudomask generation and comparison (requires torch>=2.5.1)
# RUN pip install --no-cache-dir git+https://github.com/facebookresearch/sam2.git
# RUN pip install --prefer-binary \
#         git+https://github.com/facebookresearch/sam2.git
# RUN pip install --prefer-binary \
#         git+https://github.com/facebookresearch/sam3.git

RUN git clone https://github.com/facebookresearch/sam2.git /opt/sam2 && \
    git clone https://github.com/facebookresearch/sam3.git /opt/sam3 && \
    pip install --prefer-binary /opt/sam2 /opt/sam3
    
# Install Python dependencies (segmentation_models_train)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Non-root user matching host UID/GID to avoid permission issues on mounted volume
RUN groupadd -g "${GID}" app && \
    useradd -m -u "${UID}" -g app -s /bin/bash app && \
    chown -R app:app /workspace

USER app
WORKDIR /workspace
CMD ["bash"]
