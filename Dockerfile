FROM python:3.11

WORKDIR /app


# Install OpenGL and other dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt


RUN pip3 install git+https://github.com/facebookresearch/detectron2.git

COPY api/ /app/api/
COPY checkpoints/ /app/checkpoints/
COPY data/ /app/data/
COPY dellma/ /app/dellma/
COPY model/ /app/model/
COPY util/ /app/util/
COPY runner/ /app/runner/

ENV PYTHONPATH=/app

EXPOSE 8080


CMD ["python", "api/run.py"]