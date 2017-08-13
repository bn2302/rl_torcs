#!/bin/bash

export NVIDIA_DRIVER=384.59
export NVIDIADOCKER_VERSION=1.0.1
export DOCKER_COMPOSE=1.15.0
export VGL_VERSION=2.5.2
export LIBJPEG_VERSION=1.4.2
export TURBOVNC_VERSION=2.0.1

# Install necessary ubuntu packages
apt-get update
apt-get upgrade -y
apt-get install -y \
    build-essential \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common \
    xorg

# Nvidia driver
curl -O http://us.download.nvidia.com/XFree86/Linux-x86_64/${NVIDIA_DRIVER}/NVIDIA-Linux-x86_64-${NVIDIA_DRIVER}.run
chmod +x ./NVIDIA-Linux-x86_64-*.run
./NVIDIA-Linux-x86_64-*.run -q -a -n -X -s
rm ./NVIDIA-Linux-x86_64-*.run

cp xorg.conf /etc/X11/xorg.conf

# Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

apt-get update -y
apt-get install -y docker-ce
usermod -aG docker ubuntu

# Nvidia docker
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v${NVIDIADOCKER_VERSION}/nvidia-docker_${NVIDIADOCKER_VERSION}-1_amd64.deb
dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

# Docker compose
curl -L https://github.com/docker/compose/releases/download/$(DOCKER_COMPOSE)/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
rm -f /usr/bin/docker-compose
ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

# Nvidia docker compose
apt-get install python-pip -y
pip install --upgrade pip
pip install nvidia-docker-compose

wget http://downloads.sourceforge.net/project/virtualgl/${VGL_VERSION}/virtualgl_${VGL_VERSION}_amd64.deb
dpkg -i virtualgl*.deb && rm virtualgl*.deb

# Set VirtualLG defaults, xauth bits, this adds a DRI line to xorg.conf.
#/opt/VirtualGL/bin/vglserver_config -config -s -f +t
/opt/VirtualGL/bin/vglserver_config -config +s +f -t  # access open to all users, restricting users doesn't really work :\


apt-get install -y mesa-utils

# install turbovnc
wget http://downloads.sourceforge.net/project/libjpeg-turbo/${LIBJPEG_VERSION}/libjpeg-turbo-official_${LIBJPEG_VERSION}_amd64.deb
dpkg -i libjpeg-turbo-official*.deb && rm libjpeg-turbo-official*.deb
wget http://downloads.sourceforge.net/project/turbovnc/${TURBOVNC_VERSION}/turbovnc_${TURBOVNC_VERSION}_amd64.deb
dpkg -i turbovnc*.deb && rm turbovnc*.deb

# install window manager
# installing mate as it's supported out of the box by turbovnc, see ~/.vnc/xstartup.turbovnc for more info
apt-get install mate -y --no-install-recommends

# install lightdm
apt-get install -qqy lightdm

rm /etc/lightdm/lightdm.conf
# overriding deprecated default configuration [SeatDefaults] https://wiki.ubuntu.com/LightDM
cat << EOF - > /etc/lightdm/lightdm.conf
[Seat:seat0]
display-setup-script=/usr/bin/vglgenkey
display-setup-script=xhost +LOCAL:
EOF

service lightdm start
