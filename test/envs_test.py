import gymnasium as gym

import mygym


def run_env(env):
    """Tests running roy gym envs."""
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    env.close()


def test_pickandplace():
    env = gym.make("PandaPickAndPlace-v3")
    run_env(env)


def test_dense_pickandplace():
    env = gym.make("PandaPickAndPlaceDense-v3")
    run_env(env)


def test_pickandplace_joints():
    env = gym.make("PandaPickAndPlaceJoints-v3")
    run_env(env)


def test_dense_pickandplace_joints():
    env = gym.make("PandaPickAndPlaceJointsDense-v3")
    run_env(env)

