#!/bin/bash
apt-get update

apt-get install -y xorg

apt-get install build-essential -y
curl -O http://us.download.nvidia.com/XFree86/Linux-x86_64/375.51/NVIDIA-Linux-x86_64-375.51.run
chmod +x ./NVIDIA-Linux-x86_64-*.run
./NVIDIA-Linux-x86_64-*.run -q -a -n -X -s

cat << EOF > /etc/X11/xorg.conf

Section "ServerLayout"
    Identifier     "Layout0"
    Screen      0  "Screen0"
    InputDevice    "Keyboard0" "CoreKeyboard"
    InputDevice    "Mouse0" "CorePointer"
EndSection

Section "Files"
EndSection

Section "InputDevice"
    # generated from default
    Identifier     "Mouse0"
    Driver         "mouse"
    Option         "Protocol" "auto"
    Option         "Device" "/dev/psaux"
    Option         "Emulate3Buttons" "no"
    Option         "ZAxisMapping" "4 5"
EndSection

Section "InputDevice"
    # generated from default
    Identifier     "Keyboard0"
    Driver         "kbd"
EndSection

Section "Monitor"
    Identifier     "Monitor0"
    VendorName     "Unknown"
    ModelName      "Unknown"
    HorizSync       28.0 - 33.0
    VertRefresh     43.0 - 72.0
    Option         "DPMS"
EndSection

Section "Device"
    Identifier     "Device0"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BusID          "PCI:0:3:0" # THIS MAY CHANGE FROM INSTANCE TO INSTANCE,
                               # check the device bus by running 'lspci | grep -i nvidia'
                               # or lshw -C video
EndSection

Section "Screen"
    Identifier     "Screen0"
    Device         "Device0"
    Monitor        "Monitor0"
    DefaultDepth    24
    Option         "AllowEmptyInitialConfiguration" "True" # set by the --allow-empty-initial-configuration flag
    SubSection     "Display"
        Depth       24
    EndSubSection
EndSection
EOF

# Docker
apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
apt-add-repository 'deb https://apt.dockerproject.org/repo ubuntu-xenial main'
apt-get update -y
apt-get install -y docker-engine
usermod -aG docker ubuntu

# Nvidia docker
export NVIDIADOCKER_VERSION=1.0.1
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v${NVIDIADOCKER_VERSION}/nvidia-docker_${NVIDIADOCKER_VERSION}-1_amd64.deb
dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

# Docker compose
curl -L https://github.com/docker/compose/releases/download/1.8.1/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
rm -f /usr/bin/docker-compose
ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

# Nvidia docker compose
apt-get install python-pip -y
pip install --upgrade pip
pip install nvidia-docker-compose

# install lightdm
apt-get install -qqy lightdm

service lightdm start
