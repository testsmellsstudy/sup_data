# VERSION 1.0
FROM python:3.8-slim

RUN pip install --upgrade pip

MAINTAINER Kevin Corbin, kecorbin@cisco.com, Updated by Quinn Snyder <qsnyder@cisco.com>

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    git \
    graphviz-dev \
    pkg-config \
    python3-pip \
    vim-tiny \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
 
COPY . /opt
WORKDIR /opt
RUN pip install rpds-py pytest coverage six
RUN python setup.py install 
RUN python setup.py develop
RUN apt-get clean
WORKDIR /opt
CMD ["/bin/bash"]



