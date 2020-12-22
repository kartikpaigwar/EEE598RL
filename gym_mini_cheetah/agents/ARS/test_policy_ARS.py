import sys, os
import gym
import gym_mini_cheetah
import argparse
import numpy as np

policy = np.load("./experiments/20Dec9/iterations/policy_130.npy")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--EpisodeLength', help='number of gait steps of a episode', type=int, default=1000)
	args = parser.parse_args()

	env = gym.make("mini_cheetah-v0", render= True, on_rack=False, end_steps = args.EpisodeLength)
	steps = 0
	t_r = 0

	state = env.reset()

	for i_step in range(args.EpisodeLength):

		action = policy.dot(state)
		state, r,done, [] = env.step(np.array(action))
		t_r += r

	print("Total_reward "+ str(t_r))
