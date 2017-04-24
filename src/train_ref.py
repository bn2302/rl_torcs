#!/usr/bin/python3
import docker
from ddpg import DDPG


if __name__ == '__main__':

    docker_client = docker.from_env()

    ddpg = DDPG(
        docker_client, 'ddpg_ref', 3103, '../models/ddpg_ref',
        '../logs/ddpg_ref')

    ddpg.train('g-track-1', False)
