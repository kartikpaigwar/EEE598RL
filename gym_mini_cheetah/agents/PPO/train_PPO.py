import os

import gym
import gym_mini_cheetah
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from utils.vec_monitor import VecMonitor


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'trained_models')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path + "/best_model")

        return True

# Create log dir
parent_dir = os.path.dirname(os.path.abspath(__file__))

log_dir = os.path.join(parent_dir, "experiments/24Dec1")
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = make_vec_env('mini_cheetah-v0', n_envs=4)
env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   clip_obs=10., clip_reward= 10.)

env = VecMonitor(env, log_dir)

model = PPO('MlpPolicy', env, learning_rate= 0.00025,gae_lambda=0.95,use_sde = True, n_epochs=20,
            n_steps=800, clip_range=0.2, batch_size=64, sde_sample_freq=4 ,
            tensorboard_log=log_dir)
#Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
# Train the agent
time_steps = 1000000
model.learn(total_timesteps=int(time_steps), callback=callback, tb_log_name="tb_log")
current_model_path = os.path.join(log_dir, "trained_models/current_model")
model.save(current_model_path)
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
env.save(stats_path)