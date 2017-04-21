from gym import spaces
import os
import time
import collections as col
import numpy as np
import snakeoil3_gym as snakeoil3


class TorcsDockerEnv(object):
    '''A torcs docker environment

       based on gym_torcs, here we only consider vision with throttle as
       input
    '''

    def __init__(
            self, docker_client, name="torcs", port=3101, vncport=5901,
            torcsdocker_id='bn2302/torcs'):

        self.terminal_judge_start = 100
        self.termination_limit_progress = 5
        self.default_speed = 50
        self.initial_reset = True

        self.name = name
        self.docker_client = docker_client
        self.port = port
        self.vncport = vncport

        self.torcsdocker_id = torcsdocker_id

        self.container = self._start_docker()
        self.container.exec_run("start_torcs.sh", detach=True)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf,
                         255])
        low = np.array([0., -np.inf, -np.inf, -np.inf,
                        0., -np.inf, 0., -np.inf, 0])
        self.observation_space = spaces.Box(low=low, high=high)

    def _start_docker(self):
        os.system(
            'nvidia-docker run' +
            ' --rm' +
            ' -it' +
            ' -p {:d}:5901'.format(self.vncport) +
            ' --volume="/tmp/.X11-unix/X0:/tmp/.X11-unix/X0:rw"' +
            ' --volume="/usr/lib/x86_64-linux-gnu/libXv.so.1:/usr/lib/x86_64-linux-gnu/libXv.so.1:rw"' +
            ' -p {:d}:3101/udp'.format(self.port) +
            ' --name={}'.format(self.name) +
            ' -d {}'.format(self.torcsdocker_id))

        return self.docker_client.containers.get(self.name)
           
    def reset(self, relaunch=False):

        self.time_step = 0

        if not self.initial_reset:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            if relaunch is True:
                self.container.exec_run("kill_torcs.sh", detach=True)
                self.container.exec_run("start_torcs.sh", detach=True)

        self.client = snakeoil3.Client(p=self.port)

        self.client.MAX_STEPS = np.inf

        self.client.get_servers_input()
        obs = self.client.S.d
        self.observation = self._make_observaton(obs)

        self.last_u = None

        self.initial_reset = False

        return self.get_obs()

    def end(self):
        self.container.stop()

    def step(self, u):

        # Apply Action
        this_action = self.agent_to_torcs(u)
        action_torcs = self.client.R.d
        action_torcs['steer'] = this_action['steer']
        action_torcs['accel'] = this_action['accel']
        action_torcs['brake'] = this_action['brake']

        action_torcs['gear'] = 1
        if self.client.S.d['speedX'] > 50:
            action_torcs['gear'] = 2
        if self.client.S.d['speedX'] > 80:
            action_torcs['gear'] = 3
        if self.client.S.d['speedX'] > 110:
            action_torcs['gear'] = 4
        if self.client.S.d['speedX'] > 140:
            action_torcs['gear'] = 5
        if self.client.S.d['speedX'] > 170:
            action_torcs['gear'] = 6

        # Save the privious full-obs from torcs for the reward calculation
        damage_pre = self.client.S.d["damage"]

        # One-Step Dynamics Update
        # Apply the Agent's action into torcs
        self.client.respond_to_server()
        self.client.get_servers_input()

        # Get the current full-observation from torcs
        obs = self.client.S.d
        self.observation = self._make_observaton(obs)

        # Reward setting Here
        # direction-dependent positive reward
        progress = (
            np.array(obs['speedX']) *
            (np.cos(obs['angle']) - np.sin(obs['angle'])))

        reward = progress

        # collision detection
        if obs['damage'] - damage_pre > 0:
            reward = -1

        # Episode is terminated if the agent runs backward
        if np.cos(obs['angle']) < 0:
            self.client.R.d['meta'] = True

        if self.client.R.d['meta'] is True:
            self.client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, self.client.R.d['meta'], {}

    def get_obs(self):
        return self.observation

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0], 'accel': u[1], 'brake': u[2]}

        return torcs_action

    def _make_observaton(self, raw_obs):
        names = ['focus',
                 'speedX', 'speedY', 'speedZ',
                 'angle',
                 'damage',
                 'opponents',
                 'rpm',
                 'track',
                 'trackPos',
                 'wheelSpinVel',
                 'img']
        Observation = col.namedtuple('Observation', names)
        image_rgb = self._obs_vision_to_image_rgb(raw_obs['img'])

        return Observation(
            focus=np.array(raw_obs['focus'], dtype=np.float32) / 200.,
            speedX=np.array(raw_obs['speedX'], dtype=np.float32)
            / self.default_speed,
            speedY=np.array(raw_obs['speedY'], dtype=np.float32)
            / self.default_speed,
            speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)
            / self.default_speed,
            angle=np.array(raw_obs['angle'], dtype=np.float32) / 3.1416,
            damage=np.array(raw_obs['damage'], dtype=np.float32),
            opponents=np.array(raw_obs['opponents'], dtype=np.float32) / 200.,
            rpm=np.array(raw_obs['rpm'], dtype=np.float32),
            track=np.array(raw_obs['track'], dtype=np.float32) / 200.,
            trackPos=np.array(raw_obs['trackPos'], dtype=np.float32) / 1.,
            wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
            img=image_rgb)

    def _obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec = obs_image_vec
        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (64, 64)
        r = np.array(r).reshape(sz)
        g = np.array(g).reshape(sz)
        b = np.array(b).reshape(sz)
        return np.array([r, g, b], dtype=np.uint8)


if __name__ == '__main__':
    import docker

    docker_client = docker.from_env()
    # Generate a Torcs environment
    env = TorcsDockerEnv(docker_client)
    env.reset(True)
    env.end()
