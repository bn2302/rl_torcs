# rl_torcs

## Installation

The Torcs reinforcement learning environment is based on an Amazon EC2 g2.2xlarge instance running Ubuntu 16.04. 

To setup the environment, the following commands have to be executed from a shell:

	git clone https://github.com/bn2302/rl_torcs
	cd rl_torc/docker/
	sudo su
	source root_setup.sh
	reboot

After rebooting the instance please run the following commands:

	cd rl_torc/docker
	source user_setup.sh

The script will install the nvidia drivers, nvidia-docker and an xserver, which
is used to connect to the agent via virtualgl.

Next to that the script will build the images for two docker containers:

    * Torcs running in a container with virtualgl and turbovnc
    
    * A reinforcement learning environment containing Tensorflow, a modified
      vim and other goodies

## Start the docker container
The reinforcement learning docker environment is started using `start_rl` to reattach the environment the alias `attach_rl` can be used.

## Start the training

The different agents can be trained using the scripts called  `train_X.py`.
Please not if an agent is prematurely canceled the corresponding torcs
container must be stopped using `docker stop NAME `. To list the running
containers please use  `docker ps -l -a`

## Monitor the training
To monitor the training process, please connect to the containers using the
following command

## Start the testing




