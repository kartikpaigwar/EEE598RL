import sys, os
import gym
import gym_mini_cheetah
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env_name', help='name of the gym env', type=str, default='mini_cheetah-v0')
parser.add_argument('--log_dir', help='directory to save log', type=str, default='24Dec1')
parser.add_argument('--record', help='', type=bool, default=False)
args = parser.parse_args()

args.log_dir = "./experiments/" + args.log_dir
if args.record :
    video_log_path = args.log_dir + "/run.mp4"
else:
    video_log_path = None
env = DummyVecEnv([lambda: gym.make(args.env_name, render=True, save_path=video_log_path)])
env = VecNormalize.load(args.log_dir + "/vec_normalize.pkl", env)
model = PPO.load(args.log_dir + "/models/best_model.zip")

env.training = False
env.norm_reward = False

# for i_episode in range(0,1):
obs = env.reset()
for t in range(1000):
    action, _states = model.predict(obs,deterministic=True)
    obs, rewards, done, info = env.step(action)
    if done:
        print("Episode done")
        break
env.close()

