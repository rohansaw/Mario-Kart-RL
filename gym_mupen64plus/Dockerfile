FROM ubuntu:xenial-20170915 AS base


# Setup environment variables in a single layer
ENV \
    # Prevent dpkg from prompting for user input during package setup
    DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true \
    # mupen64plus will be installed in /usr/games; add to the $PATH
    PATH=$PATH:/usr/games \
    # Set default DISPLAY
    DISPLAY=:0


################################################################
FROM base AS buildstuff

RUN apt-get update && \
    apt-get install -y \
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
    libjson-c-dev

# clone, build, and install the input bot
# (explicitly specifying commit hash to attempt to guarantee behavior within this container)
WORKDIR /src/mupen64plus-src
RUN git clone https://github.com/mupen64plus/mupen64plus-core && \
    cd mupen64plus-core && \
    git reset --hard 12d136dd9a54e8b895026a104db7c076609d11ff && \
    cd .. && \
    git clone https://github.com/kevinhughes27/mupen64plus-input-bot && \
    cd mupen64plus-input-bot && \
    git reset --hard 0a1432035e2884576671ef9777a2047dc6c717a2 && \
    make all && \
    make install


################################################################
FROM base


# Update package cache and install dependencies
RUN apt-get update && \
    apt-get install -y \
    python python-pip python-setuptools python-dev python3\
    wget \
    xvfb libxv1 x11vnc \
    imagemagick \
    mupen64plus \
    nano \
    ffmpeg \
    libjson-c2

# Upgrade pip (pip 21.0 dropped support for Python 2.7 in January 2021 - https://stackoverflow.com/a/65896996/9526448)
# TODO: Python3 upgrade - https://github.com/bzier/gym-mupen64plus/issues/81
RUN pip install --upgrade "pip < 21.0"

# Install VirtualGL (provides vglrun to allow us to run the emulator in XVFB)
# (Check for new releases here: https://github.com/VirtualGL/virtualgl/releases)
ENV VIRTUALGL_VERSION=2.5.2
RUN wget "https://sourceforge.net/projects/virtualgl/files/${VIRTUALGL_VERSION}/virtualgl_${VIRTUALGL_VERSION}_amd64.deb" && \
    apt install ./virtualgl_${VIRTUALGL_VERSION}_amd64.deb && \
    rm virtualgl_${VIRTUALGL_VERSION}_amd64.deb

# Install dependencies (here for caching)
RUN pip install \
    PyYAML==5.1 \
    termcolor==1.1.0 \
    mss==4.0.2 \
    pathlib

# Copy compiled input plugin from buildstuff layer
COPY --from=buildstuff /usr/local/lib/mupen64plus/mupen64plus-input-bot.so /usr/local/lib/mupen64plus/

# Copy the gym environment (current directory)
COPY . /src/

# Declare ROMs as a volume for mounting a host path outside the container
VOLUME /src/envs/ROMs/

WORKDIR /src

# Expose the default VNC port for connecting with a client/viewer outside the container
EXPOSE 5900
# Expose port of input-bot
# ToDo: This is hard-coded, however this might change?
EXPOSE 8082

CMD ["python", "envs/container_init.py"]