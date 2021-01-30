FROM public.ecr.aws/ubuntu/ubuntu:focal
RUN apt-get update

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

# RUN pip3 install ipython



WORKDIR /soundclassification/

COPY ./melspecs-list.py /soundclassification/melspecs/
COPY ./labels.csv /soundclassification/melspecs/

ENTRYPOINT ["python3", "/soundclassification/melspecs/melspecs-list.py"]

