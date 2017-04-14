FROM plumbee/nvidia-virtualgl

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

CMD ["/bin/bash"]