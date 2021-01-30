# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.155.1/containers/python-3-anaconda/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/vscode/devcontainers/anaconda:0-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

RUN DEBIAN_FRONTEND="noninteractive" apt-get update \
  && apt-get install -y build-essential \
  wget \
  unzip \
  bzip2 \
  software-properties-common \
  python-setuptools

# Music libraries
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y libasound-dev \
  libsndfile1 \
  portaudio19-dev \
  libportaudio2 \
  libportaudiocpp0 \
  ffmpeg \
  libavcodec-extra

RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y python3-pip

RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh 
RUN bash Anaconda3-2020.11-Linux-x86_64.sh -b -p /home/vscode/conda
ENV PATH "$PATH:/home/vscode/conda/bin"

RUN conda install llvmlite

RUN pip3 install matplotlib librosa mutagen noisereduce statistics boto3 tqdm scipy numpy pandas sklearn 

RUN pip3 install ipython
