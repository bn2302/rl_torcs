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
To monitor the training process, please connect to the containers, go into the
`logs` directory and start tensorboard using 

	tensorboard --logdir=a3c_0:'./a3c/train_0/',a3c_1:'./a3c/train_1',a3c_2:'./a3c/train_2',a3c_3:'./a3c/train_3',a3c_4:'./a3c/train_4/',a3c_5:'./a3c/train_5',a3c_6:'./a3c/train_6',a3c_7:'./a3c/train_7',ddpg_0:'./ddpg_0',ddpg_1:'./ddpg_1',dddpg_ref:'./ddpg_ref',ddpg_2:'./ddpg_2/'

Tensorboard can be accessed via port 6006 from a browser. When connecting to an
AWS instance via ssh, forward the port with `-L 6006:localhost:6006`. Then tensorboard can be opened in a browser using `http://localhost:6006/`

## Start the testing
Testing is done in the Jupyter notebook test.ipynb .
To start the jupyter server run the following command from the rl_torcs main directory

	jupyter server --allow-root

Jupyter can be accessed via port 8888 from a browser. When connecting to an
AWS instance via ssh, forward the port with `-L 6006:localhost8888` 


##References

https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb as the basis for the A3c implementation.

https://github.com/yanpanlau/DDPG-Keras-Torcs as the basis for the DDPG.

https://github.com/plumbee/nvidia-hw-accelerated-box as the basis for the setup scripts.
