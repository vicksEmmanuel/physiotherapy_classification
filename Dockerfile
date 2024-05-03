FROM python:3.11 AS build

WORKDIR /app

# Install OpenGL and other dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg

# Install Git LFS
RUN apt-get install -y git-lfs
RUN git lfs install

# Install Python dependencies
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt

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

# Install Nginx
RUN apt-get update && apt-get install -y nginx

# Copy Nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose ports
EXPOSE 80 443

# Set entry point
CMD ["sh", "-c", "python api/run.py & nginx -g 'daemon off;'"]