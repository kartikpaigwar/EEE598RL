import sys, os
import gym
import gym_mini_cheetah
import argparse
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback


class HyperParameters():
    """
    This class is basically a struct that contains all the hyperparameters that you want to tune
    """

    def __init__(self, env = "", learning_rate=0.0003, n_steps=2048, n_envs = 8, batch_size=64, n_epochs=10,
                  clip_range=0.2,  use_sde=False, sde_sample_freq=-1):
        self.env =env
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.batch_size= batch_size
        self.n_epochs = n_epochs
        self.n_envs = n_envs
        self.msg = "msg"
        self.use_sde = use_sde
        self.clip_range = clip_range
        self.sde_sample_freq = sde_sample_freq
        self.logdir = ""

    def to_text(self, path):
        res_str = ''
        res_str = res_str + 'env_name: ' + str(self.env) + '\n'
        res_str = res_str + 'learning_rate: ' + str(self.learning_rate) + '\n'
        res_str = res_str + 'n_epochs: ' + str(self.n_epochs) + '\n'
        res_str = res_str + 'n_steps: ' + str(self.n_steps) + '\n'
        res_str = res_str + 'n_envs: ' + str(self.n_envs) + '\n'
        res_str = res_str + 'use_sde: ' + str(self.use_sde) + '\n'
        res_str = res_str + 'sde_sample_frequency: ' + str(self.sde_sample_freq) + '\n'
        res_str = res_str + 'clip_range: ' + str(self.clip_range) + '\n'
        res_str = res_str + 'batch_size: ' + str(self.batch_size) + '\n'

        res_str = res_str + self.msg + '\n'
        fileobj = open(path, 'w')
        fileobj.write(res_str)
        fileobj.close()


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env_name', help='name of the gym env', type=str, default='mini_cheetah-v0')
parser.add_argument('--log_dir', help='directory to save log', type=str, default='new_log')
parser.add_argument('--lr', help='learning rate', type=float, default=0.00025)
parser.add_argument('--use_sde', help='Whether to use generalized State Dependent Exploration (gSDE)', type=bool, default=True)
parser.add_argument('--clip_range', help='clipping parameters', type=float, default=0.2)
parser.add_argument('--batch_size', help='mini batch size', type=int, default=64)
parser.add_argument('--n_steps', help='number of steps to run for each environment per update', type=int, default=800)
parser.add_argument('--n_epochs', help='number of epochs when optimizing surrogate loss', type=int, default=16)
parser.add_argument('--n_envs', help='number of env copies running in parallel', type=int, default=8)
parser.add_argument('--sde_freq', help='SDE sample frequency', type=int, default=4)

args = parser.parse_args()

hp = HyperParameters()
hp.env = args.env_name
print("Training for Environment : ", hp.env)
hp.logdir = args.log_dir
print("log dir", args.log_dir)
hp.learning_rate = args.lr
hp.use_sde = args.use_sde
hp.clip_range = args.clip_range
hp.batch_size = args.batch_size
hp.n_steps = args.n_steps
hp.n_epochs = args.n_epochs
hp.n_envs = args.n_envs
hp.sde_sample_freq =args.sde_freq

args.log_dir = "./experiments/" + args.log_dir

if os.path.isdir(args.log_dir) == False:
    os.mkdir(args.log_dir)

os.chdir(args.log_dir)
log_dir = os.getcwd()

if os.path.isdir('models') == False: os.mkdir('models')
model_save_path = log_dir + "/models/model.zip"

if os.path.isdir('logs') == False: os.mkdir('logs')
tb_log_dir = log_dir + "/logs"

os.chdir("../../")

hp.to_text('hyperparameters')

env = make_vec_env(hp.env, n_envs=args.n_envs)
env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   clip_obs=10., clip_reward= 10.)


model = PPO('MlpPolicy', env = hp.env, learning_rate=hp.learning_rate,gae_lambda=0.95,use_sde = hp.use_sde, n_epochs=hp.n_epochs,
            n_steps=hp.n_steps, clip_range=hp.clip_range, device="cuda", batch_size=hp.batch_size, sde_sample_freq=hp.sde_sample_freq,
            tensorboard_log=tb_log_dir)

checkpoint_callback = CheckpointCallback(save_freq=12800, save_path=log_dir + "/models/",
                                         name_prefix='policy')

eval_env = make_vec_env(hp.env, n_envs=1)

eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir + "/models/",
                             log_path=log_dir + "/logs/", eval_freq=6400, n_eval_episodes = 1,
                             deterministic=True, render=False)
time_steps = 1500000
model.learn(total_timesteps=int(time_steps), callback=eval_callback, tb_log_name="tensorboard_file")
model.save(model_save_path)
print("model saved at ", model_save_path)
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
env.save(stats_path)
