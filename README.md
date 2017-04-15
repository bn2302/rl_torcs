# rl_torcs

first don't use vision

should it be optional?

# Amazon EC2 

Starting point G2 instance and Ubuntu 16.04 image

Install nvidia docker and docker


 nvidia-docker run --volume="/tmp/.X11-unix/X0:/tmp/.X11-unix/X0:rw" --volume="/usr/lib/x86_64-linux-gnu/libXv.so.1:/usr/lib/x86_64-linux-gnu/libXv.so.1" -p 3101:3101/udp -it bn2302/torcs

https://github.com/plumbee/nvidia-virtualgl

nvidia-docker run -d \
     --env="DISPLAY" \
     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
     --volume="/usr/lib/x86_64-linux-gnu/libXv.so.1:/usr/lib/x86_64-linux-gnu/libXv.so.1" \
     plumbee/nvidia-virtualgl:2.5.2 vglrun glxgears
