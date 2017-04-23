import docker
import tensorflow as tf
from gym_torcs_docker import TorcsDockerEnv
from networks import ActorNetwork
from a3c import A3CNetwork, obs_to_state


def testModelOnTrack(
        docker_client, sess, model, trackname, max_steps=1000,
        docker_port=3101):

    env = TorcsDockerEnv(
        docker_client, 'test_{}_{}'.format(model, trackname), docker_port)
    obs = env.reset(relaunch=True)
    state = obs_to_state(obs)

    total_reward = 0

    for _ in range(max_steps):
        action = model.predict(state)
        obs, reward, done, _ = env.step(action)
        state = obs_to_state(obs)
        total_reward += reward

        if done:
            break



def main():
    docker_client = docker.from_env()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    test_tracks = ['']
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        pass

