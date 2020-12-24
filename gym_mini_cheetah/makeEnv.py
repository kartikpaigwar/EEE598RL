import gym
import gym_mini_cheetah
from stable_baselines3.common.env_checker import check_env

env = gym.make("mini_cheetah-v0")
check_env(env)
print("successfully created mini_cheetah-v0 gym environment")