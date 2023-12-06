ARG PYTORCH="1.5"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git libglib2.0-0 libsm6 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pytorch
RUN conda install -y pytorch torchvision cpuonly -c pytorch

# Install mmcv
RUN pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html

# Install opencv-python-headless intended for headless envts like Docker
RUN pip install opencv-python-headless --default-timeout=100

# Install mmfashion
COPY . /mmfashion
WORKDIR /mmfashion
ENV FORCE_CUDA="0"
RUN pip install -r requirements.txt --default-timeout=100
RUN pip install --no-cache-dir -e .
EXPOSE 5000
ENTRYPOINT [ "python", "-m", "flask", "--app", "api/run", "run", "--host=0.0.0.0" ]