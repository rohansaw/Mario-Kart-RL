## Setup
0. Make sure you have docker and docker compose installed
1. Build your docker image from the given docker file with `sudo docker build -t <image_name>:<tag> .` (e.g.: `sudo docker build -t mupen64_gym:one .`)
2. Create a file called `.env` and add the following line to it: `IMAGE_SPEC=<image_name>:<tag>` (e.g.: `IMAGE_SPEC=mupen64_gym`)
3. Download a MarioKart N64 ROM file (just use google) and place it in the `gym_mupen64plus/ROMs` folder with the name `marioKart.n64`

## Running the agent
1. Run `docker-compose -p <image_name> up -d`
2. If you want to see the visual output download VNCViewer and connect to the port that is exposed by docker. (Exposed port can be found with `docker ps`)

However, this currently gives no access to bash out, if you want to see the program output run `docker run -it --name <image_name> -p 5900 --mount source="$(pwd)/gym_mupen64plus/ROMs",target=/src/gym-mupen64plus/gym_mupen64plus/ROMs,type=bind <image_name>:<tag_name> python gym-mupen64plus/agent.py`

## Using stable baselines agent
Follow SB installation instructions (here)[https://stable-baselines.readthedocs.io/en/master/guide/install.html]