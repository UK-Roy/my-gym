import gymnasium as gym
import numpy as np
import time

from stable_baselines3 import DQN, PPO, SAC, TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

import mygym

env = gym.make("PandaPickAndPlace-v3", render_mode="human")
# gym.utils.play.play(env, fps=8)
# Types of Policies: CnnPolicy, MlpPolicy, MultiInputPolicy
# Instantiate the agent
model = DDPG(policy="MultiInputPolicy", env=env)
# model = DQN("MultiInputPolicy", env, verbose=1)
# model = PPO("MultiInputPolicy", env, verbose=1)
# model = SAC("MultiInputPolicy", env, train_freq=1, gradient_steps=2, verbose=1)

# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# model = TD3("MultiInputPolicy", env, action_noise=action_noise, verbose=0)

# model.train(3000)
# Train the agent and display a progress bar
model.learn(total_timesteps=int(2e5), progress_bar=True)
# Save the agent
model.save("Panda_pick&place")

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("Panda_pick&place", env=env, print_system_info=True)
# model = DQN.load("Panda_pick&place", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
print(f"Mean Reward: {mean_reward}\nSTD Reward:{std_reward}")

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
# print(obs)
# observation, info = env.reset()
# print(observation, info)
for _ in range(1000):
    action = model.predict(obs)
    # observation, reward, terminated, truncated, info = vec_env.step(action)
    obs, reward, terminated, info = vec_env.step(action)
    truncated = info[0]['TimeLimit.truncated']
    Is_success = info[0]['is_success'] 
    done = truncated or terminated
    vec_env.render("human")
    time.sleep(.01)

    if done:
        obs = vec_env.reset()

vec_env.close()