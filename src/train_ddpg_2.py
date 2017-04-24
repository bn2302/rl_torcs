#!/usr/bin/python3
import docker
from ddpg import DDPG


if __name__ == '__main__':

    docker_client = docker.from_env()

    ddpg = DDPG(
        docker_client, 'ddpg_2', 3104, '../models/ddpg_2',
        '../logs/ddpg_2')

    ddpg.train('', True)
