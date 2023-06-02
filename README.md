# Roy's-GYM :mechanical_arm:

Set of robotic environments based on PyBullet physics engine and gymnasium.

## Documentation

Wait to check out the [documentation](https://Roy's-GYM.readthedocs.io/en/latest/).

## Installation


### From source

```bash
git clone https://github.com/UK-Roy/my_gym.git
pip install -e my_gym
```

## Usage

```python
import gymnasium as gym
import mygym

env = gym.make('PandaPickAndPlace-v3', render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```


## Baselines results

Baselines results are available in [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) and the pre-trained agents in the [Hugging Face Hub](https://huggingface.co/sb3).

## Citation

Cite as

```bib
@article{roy2023gym,
  title        = {{Roy's-GYM: Open-Source Goal-Conditioned Environments for Robotic Learning}},
  author       = {Utsha Kumar Roy},
  year         = 2023,
}
```

If you find this useful you can buy me a coffee :grinning:.  
