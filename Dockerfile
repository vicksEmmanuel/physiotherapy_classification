FROM python:3.11

WORKDIR /app

# Install OpenGL and other dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

# Install Git LFS
RUN apt-get install -y git-lfs
RUN git lfs install

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt
RUN pip3 install git+https://github.com/facebookresearch/detectron2.git

# Clone the repository and fetch LFS files
RUN git clone <repository-url> /app/repo
WORKDIR /app/repo
RUN git lfs pull

# Copy the necessary files and directories
COPY api/ /app/api/
COPY data/ /app/data/
COPY dellma/ /app/dellma/
COPY model/ /app/model/
COPY util/ /app/util/
COPY runner/ /app/runner/

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["python", "api/run.py"]