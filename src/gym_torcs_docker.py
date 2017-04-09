import os
from gym import spaces
import numpy as np
import snakeoil3_gym as snakeoil3
import collections as col


class TorcsDockerEnv(object):
    """ 
       A torcs docker environment
       
       based on gym_torcs, here we only consider vision with throttle as 
       input
    """
    TORCS_DOCKER_ID = 'bn2302/torcs:gpu'
    
    def __init__(self, docker_client, name, port):

        self.name = name
        self.port = port

        os.system(('xhost + ;'
                   + 'nvidia-docker run'
                   + ' --name {}'.format(name) 
                   + ' -it' + 
                   + ' -p {}:3101/udp'.format(port)
                   + ' --device=/dev/snd:/dev/snd' 
                   + ' -v /tmp/.X11-unix:/tmp/.X11-unix:ro'
                   + ' -e DISPLAY=unix$DISPLAY'
                   + ' -d {}'.format(
                       TorcsDockerEnv.TORCS_DOCKER_ID)))
        
        self.container = docker_client.get(name)

        self.reset(True)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf,
                         255])
        low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf,
                        0])
        self.observation_space = spaces.Box(low=low, high=high)
    
        
    def reset(self, relaunch=False):
        
        self.time_step = 0

        self.client.R.d['meta'] = True
        self.client.respond_to_server()

        # TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
        if relaunch is True:
            self._reset_torcs()
            print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=self.port)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()
        
    def _reset_torcs(self):
        self.container.

    def end(self):
        self.container.stop()
        self.container.remove()

    def step(self, u):
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        action_torcs['accel'] = this_action['accel']

        #  Automatic Gear Change by Snakeoil
        action_torcs['gear'] = 1

        # Save the privious full-obs from torcs for the reward calculation
        damage_pre = client.S.d["damage"]

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self._make_observaton(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        sp = np.array(obs['speedX'])
        progress = sp * np.cos(obs['angle'])
        reward = progress

        # collision detection
        if obs['damage'] - damage_pre > 0:
            reward = -1

        # Termination judgement #########################
        # Episode is terminated if the car is out of track
        if track.min() < 0:  
            reward = -1
            client.R.d['meta'] = True

        # Episode terminates if the progress of agent is small
        if self.terminal_judge_start < self.time_step:  
            if progress < self.termination_limit_progress:
                client.R.d['meta'] = True

        # Episode is terminated if the agent runs backward
        if np.cos(obs['angle']) < 0:  
            client.R.d['meta'] = True

        if client.R.d['meta'] is True:  # Send a reset signal
            client.respond_to_server()

        self.time_step += 1
        return self.get_obs(), reward, client.R.d['meta'], {}

    def get_obs(self):
        return self.observation

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}
        torcs_action.update({'accel': u[1]})
        return torcs_action

    def _make_observaton(self, raw_obs):
        names = ['focus',
                 'speedX', 'speedY', 'speedZ',
                 'opponents',
                 'rpm',
                 'track',
                 'wheelSpinVel',
                 'img']
        Observation = col.namedtuple('Observaion', names)
        image_rgb = self._obs_vision_to_image_rgb(raw_obs[names[8]])

        return Observation(
            focus=np.array(raw_obs['focus'], dtype=np.float32) / 200.,
            speedX=np.array(raw_obs['speedX'], dtype=np.float32) 
                / self.default_speed,
            speedY=np.array(raw_obs['speedY'], dtype=np.float32) 
                / self.default_speed,
            speedZ=np.array(raw_obs['speedZ'], dtype=np.float32) 
                / self.default_speed,
            opponents=np.array(raw_obs['opponents'], dtype=np.float32) / 200.,
            rpm=np.array(raw_obs['rpm'], dtype=np.float32),
            track=np.array(raw_obs['track'], dtype=np.float32) / 200.,
            wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
            img=image_rgb)
  
    def _obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec = obs_image_vec
        rgb = []
        temp = []
        # convert size 64x64x3 = 12288 to 64x64=4096 2-D list 
        # with rgb values grouped together.
        # Format similar to the observation in openai gym
        for i in range(0, 12286, 3):
            temp.append(image_vec[i])
            temp.append(image_vec[i + 1])
            temp.append(image_vec[i + 2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)
