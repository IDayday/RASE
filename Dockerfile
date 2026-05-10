FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV D4RL_SUPPRESS_IMPORT_ERROR=1
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git ca-certificates build-essential curl unzip libgl1-mesa-dev \
    libgl1-mesa-glx libglew-dev libosmesa6-dev libglfw3 libglfw3-dev \
    libxrender1 libxext6 patchelf swig \
    && rm -rf /var/lib/apt/lists/*

# Miniconda.
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

WORKDIR /workspace/rase_phase0
COPY environment.yml /workspace/rase_phase0/environment.yml
RUN conda env create -f environment.yml && conda clean -afy
SHELL ["/bin/bash", "-lc"]

# MuJoCo 2.1.0 for mujoco-py / D4RL control environments.
RUN mkdir -p /root/.mujoco \
    && wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz \
    && tar -xzf /tmp/mujoco210.tar.gz -C /root/.mujoco \
    && rm /tmp/mujoco210.tar.gz
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

COPY . /workspace/rase_phase0
RUN echo "conda activate rase" >> /root/.bashrc

CMD ["bash"]
