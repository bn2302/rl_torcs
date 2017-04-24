#!/usr/bin/python3
import docker
from ddpg import DDPG


if __name__ == '__main__':

    docker_client = docker.from_env()

    ddpg = DDPG(
        docker_client, 'ddpg_0', 3101, '../models/ddpg_0',
        '../logs/ddpg_0')

    ddpg.train('', False)
