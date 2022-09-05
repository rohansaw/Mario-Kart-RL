################################################################
FROM ubuntu:focal-20220801 AS base


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

RUN apt update && \
	apt install -y --no-install-recommends \
	build-essential dpkg-dev \
        git \
        python3 python3-pip python3-setuptools python3-dev \
        wget \
        xvfb libxv1 x11vnc \
        imagemagick \
        mupen64plus-data mupen64plus-ui-console \
        nano \
        unzip \
        ffmpeg \
        libjpeg-dev libtiff-dev libgtk2.0-dev \
        libsdl2-dev libnotify-dev freeglut3 freeglut3-dev \
        libjson-c-dev

# clone, build, and install the input bot
# (explicitly specifying commit hash to attempt to guarantee behavior within this container)
WORKDIR /src/d-src
RUN git clone https://github.com/mupen64plus/mupen64plus-core && \
	cd mupen64plus-core && \
	git reset --hard 12d136dd9a54e8b895026a104db7c076609d11ff && \
	cd .. && \
	git clone https://github.com/Snagnar/mupen64plus-input-bot && \
	cd mupen64plus-input-bot && \
	git checkout feature/image-retrieval && \
	make all && \
	make install

WORKDIR /src/code

################################################################
FROM base


# Update package cache and install dependencies
RUN apt update && \
	apt install -y --no-install-recommends \
	build-essential dpkg-dev \
        git \
        python3 python3-pip python3-setuptools python3-dev \
        wget \
        xvfb libxv1 x11vnc \
        imagemagick \
        mupen64plus-data mupen64plus-ui-console \
        nano \
        unzip \
        ffmpeg \
        libjpeg-dev libtiff-dev libgtk2.0-dev \
        libsdl2-dev libnotify-dev freeglut3 freeglut3-dev \
        libjson-c-dev


# Install VirtualGL (provides vglrun to allow us to run the emulator in XVFB)
# (Check for new releases here: https://github.com/VirtualGL/virtualgl/releases)
ENV VIRTUALGL_VERSION=2.5.2
RUN wget "https://sourceforge.net/projects/virtualgl/files/${VIRTUALGL_VERSION}/virtualgl_${VIRTUALGL_VERSION}_amd64.deb" && \
	apt install ./virtualgl_${VIRTUALGL_VERSION}_amd64.deb && \
	rm virtualgl_${VIRTUALGL_VERSION}_amd64.deb



# Copy compiled input plugin from buildstuff layer
COPY --from=buildstuff /usr/local/lib/mupen64plus/mupen64plus-input-bot.so /usr/local/lib/mupen64plus/

RUN mkdir /roms

RUN mkdir /mupen-plugin
VOLUME /mupen-plugin

RUN wget https://archive.org/download/mario-kart-64-usa/Mario%20Kart%2064%20%28USA%29.zip -O /tmp/marioKart.zip && \
        unzip /tmp/marioKart.zip -d /roms && \
        mv "/roms/Mario Kart 64 (USA).n64" /roms/marioKart.n64

# Expose the default VNC port for connecting with a client/viewer outside the container
EXPOSE 8082

