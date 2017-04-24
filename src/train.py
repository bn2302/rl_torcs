#!/usr/bin/python3
import argparse
import docker
from ddpg import DDPG
from a3c import A3C


parser = argparse.ArgumentParser(description='Run commands')

parser.add_argument(
    '-a', '--algorithm', type=str, default='ddpg',
    help='Select the algorithm used for training. Choices are: ddpg and a3c')

parser.add_argument(
    '-l', '--logdir', type=str, default='../logs/ddpg',
    help='Log directory path')

parser.add_argument(
    '-m', '--modeldir', type=str, default='../models/ddpg',
    help='Model directory path')

parser.add_argument(
    '-t', '--track', type=str, default='',
    help='Track used for training, if left blank all 6 training tracks will be used')

parser.add_argument(
    '-r', '--reset', type=bool, default=True,
    help='terminate episode if stuck')

parser.add_argument(
    '-p', '--port', type=int, default=3101,
    help='terminate episode if stuck')

parser.add_argument(
    '-n', '--name', type=str, default='worker',
    help='terminate episode if stuck')


parser.add_argument(
    '-w', '--workers', type=int, default=1,
    help='terminate episode if stuck')


if __name__ == '__main__':
    args = parser.parse_args()

    docker_client = docker.from_env()

    if args.algorithm == 'ddpg':
        ddpg = DDPG(
            docker_client, args.name, args.port, args.modeldir, args.logdir)
        ddpg.train(args.track, args.reset)

    if args.algorithm == 'a3c':
        a3c = A3C(docker_client, args.port, args.modeldir, args.logdir)
        a3c.train(args.workers)
