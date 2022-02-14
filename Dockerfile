FROM ubuntu:focal

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         language-pack-ja \
         libjpeg-dev \
         libpng-dev \
         python3-dev \
         python-numpy \
         wget &&\
     rm -rf /var/lib/apt/lists/*

RUN cd /tmp && wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py

WORKDIR /app

RUN locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8

COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY app.py .
COPY checker_pytorch.py .

ENTRYPOINT [ "python3", "-u" ]
CMD [ "app.py" ]
