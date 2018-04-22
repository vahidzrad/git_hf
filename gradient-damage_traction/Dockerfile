FROM quay.io/fenicsproject/stable:2017.1.0
USER root
RUN apt-get -qq update && \
    apt-get -y upgrade && \
    apt-get -y install gmsh && \
    apt-get clean && \
    pip install h5py && \
    pip3 install h5py && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
USER root