import gymnasium as gym
import time
import mygym

env = gym.make("PandaPickAndPlace-v3", render_mode="human")
# gym.utils.play.play(env, fps=100)

observation, info = env.reset()

for _ in range(10000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    # truncated = info[0]['TimeLimit.truncated']
    # Is_success = info[0]['is_success'] 
    time.sleep(.1)
    done = truncated or terminated
    if done:
        observation, info = env.reset()

env.close()
