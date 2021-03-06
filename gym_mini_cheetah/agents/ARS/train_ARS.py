import sys, os
import inspect

# Importing the libraries
import os
import gym
import gym_mini_cheetah
import time
import multiprocessing as mp
from multiprocessing import Process, Pipe
import argparse
from utils.logger import DataLog
from utils.make_train_plots import make_train_plots_ars
import random

import numpy as np

class HyperParameters():
    """
    This class is basically a struct that contains all the hyperparameters that you want to tune
    """

    def __init__(self, action_dim=4, random_init=True, msg='', nb_steps=10000,
                 episode_length=800, learning_rate=0.02, nb_directions=16, nb_best_directions=8, noise=0.03, seed=1,
                 env_name='mini_cheetah-v0'):
        self.nb_steps = nb_steps
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.nb_directions = nb_directions
        self.nb_best_directions = nb_best_directions
        assert self.nb_best_directions <= self.nb_directions
        self.noise = noise
        self.seed = seed
        self.env_name = env_name
        self.msg = msg
        self.action_dim = action_dim
        self.random_init = random_init
        self.logdir = ""

    def to_text(self, path):
        res_str = ''
        res_str = res_str + 'env_name: ' + str(self.env_name) + '\n'
        res_str = res_str + 'learning_rate: ' + str(self.learning_rate) + '\n'
        res_str = res_str + 'noise: ' + str(self.noise) + '\n'
        res_str = res_str + 'episode_length: ' + str(self.episode_length) + '\n'
        res_str = res_str + 'direction ratio: ' + str(self.nb_directions / self.nb_best_directions) + '\n'
        res_str = res_str + 'policy initialization: ' + str(self.random_init) + '\n'

        res_str = res_str + self.msg + '\n'
        fileobj = open(path, 'w')
        fileobj.write(res_str)
        fileobj.close()


# Multiprocess Exploring the policy on one specific direction and over one episode

_RESET = 1
_CLOSE = 2
_EXPLORE = 3


def ExploreWorker(rank, childPipe, envname, args):
    env = gym.make(envname)
    nb_inputs = env.observation_space.sample().shape[0]
    observation_n = env.reset()
    n = 0
    while True:
        n += 1
        try:
            # Only block for short times to have keyboard exceptions be raised.
            if not childPipe.poll(0.001):
                continue
            message, payload = childPipe.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if message == _RESET:
            observation_n = env.reset()
            childPipe.send(["reset ok"])
            continue
        if message == _EXPLORE:
            policy = payload[0]
            hp = payload[1]
            direction = payload[2]
            delta = payload[3]
            state = env.reset()
            done = False
            num_plays = 0.
            sum_rewards = 0
            while num_plays < hp.episode_length:
                action = policy.evaluate(state, delta, direction, hp)
                state, reward, done, _ = env.step(action)
                if done:
                    break
                sum_rewards += reward
                num_plays += 1
            childPipe.send([sum_rewards, num_plays])
            continue
        if message == _CLOSE:
            childPipe.send(["close ok"])
            break
    childPipe.close()


# Building the AI

class Policy():

    def __init__(self, input_size, output_size, env_name, random_init, args):
        try:
            self.theta = np.load(args.policy)
        except:
            if (random_init):
                print("Training from random policy")
                self.theta = np.random.randn(output_size, input_size)
            else:
                print("Training from zero matrix policy")
                self.theta = np.zeros((output_size, input_size))

        self.env_name = env_name
        print("Starting policy theta=", self.theta)

    def evaluate(self, input, delta, direction, hp):
        if direction is None:
            return np.clip(self.theta.dot(input), -1.0, 1.0)
        elif direction == "positive":
            return np.clip((self.theta + hp.noise * delta).dot(input), -1.0, 1.0)
        else:
            return np.clip((self.theta - hp.noise * delta).dot(input), -1.0, 1.0)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]

    def update(self, rollouts, sigma_r, args):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, direction in rollouts:
            step += (r_pos - r_neg) * direction
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step
        timestr = time.strftime("%Y%m%d-%H%M%S")


# Exploring the policy on one specific direction and over one episode

def explore(env, policy, direction, delta, hp):
    nb_inputs = env.observation_space.sample().shape[0]
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while num_plays < hp.episode_length:
        action = policy.evaluate(state, delta, direction, hp)
        state, reward, done, _ = env.step(action)
        if done:
            break
        sum_rewards += reward
        num_plays += 1
    return sum_rewards


def policyevaluation(env, policy, hp):

    reward_evaluation = 0
    reward_evaluation = reward_evaluation + explore(env, policy, None, None, hp)

    return reward_evaluation


# Training the AI
def train(env, policy, hp, parentPipes, args):
    args.logdir = "experiments/" + args.logdir
    logger = DataLog()
    total_steps = 0
    best_return = -99999999

    working_dir = os.getcwd()

    if os.path.isdir(args.logdir) == False:
        os.mkdir(args.logdir)

    previous_dir = os.getcwd()

    os.chdir(args.logdir)
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') == False: os.mkdir('logs')
    hp.to_text('hyperparameters')

    log_dir = os.getcwd()
    os.chdir(working_dir)

    for step in range(hp.nb_steps):

        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions
        if (parentPipes):
            process_count = len(parentPipes)
        if parentPipes:
            p = 0
            while (p < hp.nb_directions):
                temp_p = p
                n_left = hp.nb_directions - p  # Number of processes required to complete the search
                for k in range(min([process_count, n_left])):
                    parentPipe = parentPipes[k]
                    parentPipe.send([_EXPLORE, [policy, hp, "positive", deltas[temp_p]]])
                    temp_p = temp_p + 1
                temp_p = p
                for k in range(min([process_count, n_left])):
                    positive_rewards[temp_p], step_count = parentPipes[k].recv()
                    total_steps = total_steps + step_count
                    temp_p = temp_p + 1
                temp_p = p

                for k in range(min([process_count, n_left])):
                    parentPipe = parentPipes[k]
                    parentPipe.send([_EXPLORE, [policy, hp, "negative", deltas[temp_p]]])
                    temp_p = temp_p + 1
                temp_p = p

                for k in range(min([process_count, n_left])):
                    negative_rewards[temp_p], step_count = parentPipes[k].recv()
                    total_steps = total_steps + step_count
                    temp_p = temp_p + 1
                p = p + process_count
                print('total steps till now: ', total_steps, 'Processes done: ', p)

        else:
            # Getting the positive rewards in the positive directions
            for k in range(hp.nb_directions):
                positive_rewards[k] = explore(env, policy, "positive", deltas[k], hp)

            # Getting the negative rewards in the negative/opposite directions
            for k in range(hp.nb_directions):
                negative_rewards[k] = explore(env, policy, "negative", deltas[k], hp)

        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {
            k: max(r_pos, r_neg)
            for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))
        }
        order = sorted(scores.keys(), key=lambda x: -scores[x])[:int(hp.nb_best_directions)]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array([x[0] for x in rollouts] + [x[1] for x in rollouts])
        sigma_r = all_rewards.std()  # Standard deviation of only rewards in the best directions is what it should be
        # Updating our policy
        policy.update(rollouts, sigma_r, args)
        #global reward_evaluation                 #change made
        reward_evaluation = policyevaluation(env, policy, hp)
        logger.log_kv('steps', step)
        logger.log_kv('return', reward_evaluation)
        if (reward_evaluation > best_return):
            best_policy = policy.theta
            best_return = reward_evaluation
            np.save(log_dir + "/iterations/best_policy.npy", best_policy)
        print('Step:', step, 'Reward:', reward_evaluation)
        policy_path = log_dir + "/iterations/" + "policy_" + str(step)
        np.save(policy_path, policy.theta)

        logger.save_log(log_dir + "/logs/")
        make_train_plots_ars(log=logger.log, keys=['steps', 'return'], save_loc=log_dir + "/logs/")


# Running the main code


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='Gym environment name', type=str, default='mini_cheetah-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1234123)
    parser.add_argument('--n_envs', help='number of parallel agents', type=int, default=8)
    parser.add_argument('--render', help='OpenGL Visualizer', type=int, default=0)
    parser.add_argument('--steps', help='Number of steps', type=int, default=500)
    parser.add_argument('--policy', help='Starting policy file (npy)', type=str, default='')
    parser.add_argument('--logdir', help='Directory root to log policy files (npy)', type=str, default='logdir_name')
    parser.add_argument('--mp', help='Enable multiprocessing', type=int, default=1)
    # these you have to set
    parser.add_argument('--lr', help='learning rate', type=float, default=0.05)
    parser.add_argument('--noise', help='noise hyperparameter', type=float, default=0.03)
    parser.add_argument('--episode_length', help='length of each episode', type=float, default=800)
    parser.add_argument('--random_init', help='random_policy_initialization', type=int, default=0)
    parser.add_argument('--msg', help='msg to save in a text file', type=str, default='')
    parser.add_argument('--action_dim', help='action space dimension', type=int,default=4)
    parser.add_argument('--directions', help='divising factor of total directions to use', type=int, default=2)
    args = parser.parse_args()


    hp = HyperParameters()
    args.policy = './initial_policies/' + args.policy
    hp.msg = args.msg
    hp.env_name = args.env
    print("\n\n", hp.env_name, "\n\n")
    env = gym.make(hp.env_name, render = False)
    hp.seed = args.seed
    hp.n_envs  = args.n_envs
    hp.nb_steps = args.steps
    hp.learning_rate = args.lr
    hp.noise = args.noise
    hp.episode_length = args.episode_length
    hp.nb_directions = int(env.observation_space.sample().shape[0] * env.action_space.sample().shape[0])
    hp.nb_best_directions = int(hp.nb_directions / args.directions)
    hp.random_init = args.random_init
    hp.action_dim = args.action_dim
    print("log dir", args.logdir)
    hp.logdir = args.logdir
    np.random.seed(hp.seed)
    max_processes = hp.n_envs
    parentPipes = None
    if args.mp:
        num_processes = min([hp.nb_directions, max_processes])
        print('processes: ', num_processes)
        processes = []
        childPipes = []
        parentPipes = []

        for pr in range(num_processes):
            parentPipe, childPipe = Pipe()
            parentPipes.append(parentPipe)
            childPipes.append(childPipe)

        for rank in range(num_processes):
            p = mp.Process(target=ExploreWorker, args=(rank, childPipes[rank], hp.env_name, args))
            p.start()
            processes.append(p)

    nb_inputs = env.observation_space.sample().shape[0]
    nb_outputs = env.action_space.sample().shape[0]
    policy = Policy(nb_inputs, nb_outputs, hp.env_name, hp.random_init, args)
    print("start training")

    train(env, policy, hp, parentPipes, args)

    if args.mp:
        for parentPipe in parentPipes:
            parentPipe.send([_CLOSE, "pay2"])

        for p in processes:
            p.join()
