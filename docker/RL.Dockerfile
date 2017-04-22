FROM tensorflow/tensorflow:1.1.0-rc2-gpu-py3

MAINTAINER Bastian Niebel  <bastian.niebel@gmail.com>

RUN apt update && apt install -y --no-install-recommends \
	apt-utils \
	curl \
	apt-transport-https \ 
	aufs-tools \
        git  \ 
	wget \
	cmake \
	build-essential \
	vim \
	less \
	htop \
	ctags \
	tmux

# docker
RUN curl -fsSL "https://download.docker.com/linux/ubuntu/gpg" | apt-key add - 
RUN add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_r\elease -cs) \
   stable"
RUN apt update && apt install -y --no-install-recommends docker-ce 
RUN pip3 install docker

# nvidia-docker
RUN wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb && \
	dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

RUN git clone https://github.com/openai/gym.git && \
    cd gym && \
    pip3 install -e .

RUN git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim


COPY vimrc1 /root/.vimrc
RUN vim +PluginInstall +qall
RUN cd /root/.vim/bundle/YouCompleteMe && python3 install.py && cd /root

COPY vimrc /root/.vimrc

RUN pip3 install flake8 pylint pyflakes pytest seaborn

RUN mkdir /root/rl_torcs 
WORKDIR "/root/rl_torcs"

CMD ["/bin/bash"]
