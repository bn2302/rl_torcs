#!/bin/bash

docker build -f Torcs.Dockerfile -t bn2302/torcs .

docker build -f RL.Dockerfile -t bn2302/rl_tf .

touch ~/.Xauthority

cat << EOF - > /home/ubuntu/.bashrc
alias start_rl="nvidia-docker run --name=rl_tf -it -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/nvidia-modprobe:/usr/bin/nvidia-modprobe -v /home/ubuntu/rl_torcs:/root/rl_torcs -m=10G --net=host --rm bn2302/rl_tf"
alias attach_rl="docker attach rl_tf"

EOF 


