import gymnasium as gym

from stable_baselines3 import DQN, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

import mygym

env = gym.make("PandaPickAndPlace-v3", render_mode="human")

# Types of Policies: CnnPolicy, MlpPolicy, MultiInputPolicy
# Instantiate the agent
model = DQN("MlpPolicy", env, verbose=1)
# model = PPO("MlpPolicy", env, verbose=1)
# model = SAC("MlpPolicy", env, train_freq=1, gradient_steps=2, verbose=1)

# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=0)

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
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()

observation, info = env.reset()

for _ in range(1000):
    action = model.predict(obs, deterministic=True)
    observation, reward, terminated, truncated, info = vec_env.step(action)
    vec_env.render("human")

    if terminated or truncated:
        observation, info = vec_env.reset()

env.close()