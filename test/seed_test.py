import gymnasium as gym
import numpy as np

import mygym


def test_seed_pick_and_place():
    final_observations = []
    env = gym.make("PandaPickAndPlace-v3")
    actions = [
        np.array([0.429, -0.287, 0.804, -0.592]),
        np.array([0.351, -0.136, 0.296, -0.223]),
        np.array([-0.187, 0.706, -0.988, 0.972]),
        np.array([-0.389, -0.249, 0.374, -0.389]),
        np.array([-0.191, -0.297, -0.739, 0.633]),
        np.array([0.093, 0.242, -0.11, -0.949]),
    ]
    for _ in range(2):
        env.reset(seed=794512)
        for action in actions:
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset()
        final_observations.append(observation)

    assert np.allclose(final_observations[0]["observation"], final_observations[1]["observation"])
    assert np.allclose(final_observations[0]["achieved_goal"], final_observations[1]["achieved_goal"])
    assert np.allclose(final_observations[0]["desired_goal"], final_observations[1]["desired_goal"])
