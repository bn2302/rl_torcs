#!/usr/bin/python3
import docker
from ddpg import DDPG


if __name__ == '__main__':

    docker_client = docker.from_env()

    ddpg = DDPG(
        docker_client, 'ddpg_1', 3102, '../models/ddpg_1',
        '../logs/ddpg_1')

    ddpg.train('', True)
