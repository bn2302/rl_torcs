FROM plumbee/nvidia-virtualgl

MAINTAINER Bastian Niebel  <bastian.niebel@gmail.com>

ENV TURBOVNC_VERSION 2.1.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    xserver-xorg \
    xauth \
    xvfb \
    lxde-core \
    lxterminal \
    openjdk-8-jre \
    icedtea-8-plugin 

# install TurboVNCL
RUN curl -sSL https://downloads.sourceforge.net/project/turbovnc/"${TURBOVNC_VERSION}"/turbovnc_"${TURBOVNC_VERSION}"_amd64.deb -o turbovnc_"${TURBOVNC_VERSION}"_amd64.deb && \
    dpkg -i turbovnc_*_amd64.deb && \
    rm turbovnc_*_amd64.deb
ENV PATH /opt/TurboVNC/bin:${PATH}

ENV DISPLAY :1

RUN mkdir ~/.vnc/ && \
    echo docker | vncpasswd -f > ~/.vnc/passwd && \
    chmod 600 ~/.vnc/passwd

RUN touch ~/.Xauthority


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
    swig \
    && rm -rf /var/lib/apt/lists/*


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

RUN apt-get update && apt-get install -y \
	python3

COPY set_track.py /usr/local/bin
RUN chmod +x /usr/local/bin/set_track.py

CMD ["/bin/bash"]
