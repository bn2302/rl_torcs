FROM tensorflow/tensorflow:1.1.0-rc2-gpu-py3

MAINTAINER Bastian Niebel  <bastian.niebel@gmail.com>

RUN apt update && apt install -y --no-install-recommends \
	apt-utils \
	curl \
	apt-transport-https \ 
	aufs-tools \
        git  \ 
	wget

# docker
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - 
RUN add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_r\elease -cs) \
   stable"
RUN apt update && apt install -y --no-install-recommends docker-ce 
RUN pip3 install docker

# nvidia-docker
RUN wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb && \
	dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

EXPOSE 3101-3199/udp

CMD ["/bin/bash"]

