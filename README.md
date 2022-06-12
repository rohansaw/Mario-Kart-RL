## Setup
0. Make sure you have docker and docker compose installed
1. Build your docker image from the given docker file with `docker build -t <image_name>:<tag> .`
2. Create a file called `.env` and add the following line to it: `IMAGE_SPEC=<image_name>:<tag>`

## Running the agent
1. Run `docker-compose -p <image_name> up -d`
2. If you want to see the visual output download VNCViewer and connect to the port that is exposed by docker. (Exposed port can be found with `docker ps`)

