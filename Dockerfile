FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /app

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV APP_ROOT /app
ENV WORK_DIR /app/workspace

ENV DEBIAN_FRONTEND noninteractive

RUN mkdir -p $APP_ROOT
WORKDIR $APP_ROOT

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update && apt-get install -y \
    build-essential \
    sudo \
    git \
    wget \
    curl \
    cmake \
    file \
    unzip \
    gcc \
    g++ \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install -U pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY trainer_qa.py /app/trainer_qa.py
COPY utils_qa.py /app/utils_qa.py
COPY run_qa_beam_search.py /app/run_qa_beam_search.py
COPY run.sh /app/run.sh

CMD ["/app/run.sh"]