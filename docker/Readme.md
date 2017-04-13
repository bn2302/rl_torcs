# Build the image
put in build command

	docker build -t bn2302/torcs:gpu -f Dockerfile .

# Test the image
Run in 

	xhost + ;nvidia-docker run -it -p 3101:3101/udp --device=/dev/snd:/dev/snd -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY -d bn2302/torcs:gpu 

use snakeoil client

	python3 snakeoil3_gym.py
	
now you should see the car moving and driving


# TODO: Run the image from python

in order to allow access for xhost, is run the docker image direcly form python
gving it a name and then launching torcs from it, in this way we can easliy 
access via the docker sdk 