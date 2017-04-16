FROM bn2302/turbovnc

MAINTAINER Bastian Niebel  <bastian.niebel@gmail.com>


RUN apt-get update && apt-get install -y \
    libglib2.0-dev \
    libgl1-mesa-dev \
    libpng12-dev \
    git \
    freeglut3-dev \
    libplib-dev \
    libopenal-dev \
    libpng12-dev \
    zlib1g-dev \
    libogg-dev \
    libvorbis-dev \
    g++ \
    libalut-dev \
    libxi-dev \
    libxmu-dev \
    libxrandr-dev \
    make \
    patch \
    xautomation  \
    libopenblas-dev \
    zlib1g-dev \
    libjpeg-dev \
    xvfb \
    libav-tools \
    xorg-dev \
    libboost-all-dev \
    libsdl2-dev \
    swig


WORKDIR "/root"
RUN git clone https://github.com/ugo-nama-kun/gym_torcs && \
    cd gym_torcs/vtorcs-RL-color && \
    ./configure && \
    make && \
    make install && \
    make datainstall && \
    cd /root && \
    rm -r gym_torcs


COPY start_torcs.sh /usr/local/bin
RUN chmod +x /usr/local/bin/start_torcs.sh

COPY kill_torcs.sh /usr/local/bin
RUN chmod +x /usr/local/bin/kill_torcs.sh

# set to no sound
COPY sound.xml /usr/local/share/games/torcs/config

RUN echo 'exec vncserver&' >> ~/.bashrc

CMD ["/bin/bash"]


