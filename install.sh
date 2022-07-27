#!/bin/bash

# if [[ $UID != 0 ]]; then
#     echo "Please run this script with sudo:"
#     echo "sudo $0 $*"
#     exit 1
# fi


# Prevent dpkg from prompting for user input during package setup
DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true \
# mupen64plus will be installed in /usr/games; add to the $PATH
PATH=$PATH:/usr/games \
# Set default DISPLAY
VIRTUALGL_VERSION=2.5.2
DISPLAY=:0

# install required packages
sudo apt-get update && \
sudo apt-get install -y \
        build-essential dpkg-dev \
        git \
        python3 python3-pip python3-setuptools python3-dev \
        wget \
        xvfb libxv1 x11vnc \
        imagemagick \
        mupen64plus-data mupen64plus-ui-console \
        nano \
        ffmpeg \
        libjpeg-dev libtiff-dev libgtk2.0-dev \
        libsdl2-dev libnotify-dev freeglut3 freeglut3-dev \
        libjson-c-dev  &&

# # clone, build, and install the input bot
# # (explicitly specifying commit hash to attempt to guarantee behavior within this container)
sudo rm -fr install &&
mkdir install &&
cd install &&
git clone https://github.com/mupen64plus/mupen64plus-core.git && \
    cd mupen64plus-core && \
    git reset --hard 12d136dd9a54e8b895026a104db7c076609d11ff && \
cd .. && \
git clone git@github.com:Snagnar/mupen64plus-input-bot.git && \
# git clone https://github.com/snagnar/mupen64plus-input-bot && \
    cd mupen64plus-input-bot && \
make all && \
sudo make install &&
cd .. &&

# # Install VirtualGL (provides vglrun to allow us to run the emulator in XVFB)
# # (Check for new releases here: https://github.com/VirtualGL/virtualgl/releases)
wget "https://sourceforge.net/projects/virtualgl/files/${VIRTUALGL_VERSION}/virtualgl_${VIRTUALGL_VERSION}_amd64.deb" && \
    sudo apt install ./virtualgl_${VIRTUALGL_VERSION}_amd64.deb && \
    rm virtualgl_${VIRTUALGL_VERSION}_amd64.deb &&
cd .. &&

sudo apt install virtualenv &&
virtualenv venv &&
. venv/bin/activate &&
pip install -e .
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

