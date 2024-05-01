FROM python:3.11 AS build
WORKDIR /app

# Install OpenGL and other dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg

# Install Git LFS
RUN apt-get install -y git-lfs
RUN git lfs install

# Install required packages
# RUN apt-get update && \
#     apt-get install -y software-properties-common && \
#     rm -rf /var/lib/apt/lists/*

# Install Python 3.11
# RUN apt-get update && \
#     apt-get install -y python3.11 && \
#     rm -rf /var/lib/apt/lists/*

# Add NVIDIA package repository
# RUN apt-get update && \
#     apt-get install -y gnupg2 curl && \
#     curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub | apt-key add - && \
#     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64 /" > /etc/apt/sources.list.d/cuda.list


# # Install CUDA components
# RUN apt-get update && apt-get install -y \
#     cuda-11-8 \
#     cuda-libraries-11-8 \
#     cuda-nvml-dev-11-8 \
#     cuda-command-line-tools-11-8 \
#     libnvidia-compute-11-8 \
#     libnvidia-decode-11-8 \
#     libnvidia-encode-11-8

# Install additional CUDA libraries (if needed)
# RUN sudo apt install ./<filename.deb>
# RUN sudo cp /var/cudnn-<something>.gpg /usr/share/keyrings/
# RUN sudo apt update
# RUN sudo apt install libcudnn8 libcudnn8-dev libcudnn8-samples

# Install Python package manager
# RUN apt-get install -y python3-pip

# Install Python dependencies
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt
RUN pip3 install git+https://github.com/facebookresearch/detectron2.git

# Create data directory
RUN mkdir -p /app/data

# Copy the necessary files and directories
COPY api/ /app/api/
COPY checkpoints/ /app/checkpoints/
COPY data/test_data/ /app/data/test_data/
COPY data/actions_dataset/ /app/data/actions_dataset/
COPY dellma/ /app/dellma/
COPY model/ /app/model/
COPY util/ /app/util/
COPY runner/ /app/runner/

# Set environment variable
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Set entry point
CMD ["python", "api/run.py"]