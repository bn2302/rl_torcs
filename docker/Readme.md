# Build the image
put in build command

	docker build -t bn2302/turbovnc -f TurboVNC.Dockerfile .
	docker build -t bn2302/torcs -f Torcs.Dockerfile .

# 

	nvidia-docker run \
		--volume="/tmp/.X11-unix/X0:/tmp/.X11-unix/X0:rw" 
		--volume="/usr/lib/x86_64-linux-gnu/libXv.so.1:/usr/lib/x86_64-linux-gnu/libXv.so.1" \
		-p 3101:3101/udp \ 
		-p 5901:5901 \
		-p 5801:5801 \
		-ti -rm \
		bn2302/torcs

	nvidia-docker run -it -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/nvidia-modprobe:/usr/bin/nvidia-modprobe -m=10G --net=host --rm bn2302/rl_tf

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
