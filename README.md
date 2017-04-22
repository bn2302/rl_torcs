# rl_torcs

## Installation

The Torcs reinforcement learning environment is based on an Amazon EC2 g2.2xlarge instance running Ubuntu 16.04. 

To setup the environment, the following commands have to be executed from a shell:

	git clone https://github.com/bn2302/rl_torcs
	cd rl_torc/docker/
	sudo su
	source root_setup.sh

	reboot

	# login
	cd rl_torc/docker
	source user_setup.sh

## Startup
The reinforcement learning environment is started using `start_rl` to reattach the environment the alias `attach_rl` can be used.


