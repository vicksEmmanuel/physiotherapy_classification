FROM python:3.11 AS build

WORKDIR /app

# Install OpenGL and other dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg

# Install Git LFS
RUN apt-get install -y git-lfs
RUN git lfs install

# Install required packages
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# Add the deadsnakes PPA
# RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.11
RUN apt-get update && \
    apt-get install -y python3.11 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --timeout 300 -r requirements.txt
RUN pip3 install git+https://github.com/facebookresearch/detectron2.git

RUN mkdir -p /app/data


# Copy the necessary files and directories from the GitHub Actions workspace
COPY api/ /app/api/
COPY checkpoints/ /app/checkpoints/
COPY data/test_data/ /app/data/test_data/
COPY data/actions_dataset/ /app/data/actions_dataset/
COPY dellma/ /app/dellma/
COPY model/ /app/model/
COPY util/ /app/util/
COPY runner/ /app/runner/

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["python", "api/run.py"]