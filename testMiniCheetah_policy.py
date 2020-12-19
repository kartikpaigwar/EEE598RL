import sys, os
#sys.path.append(os.path.realpath('../')) False, True
import gym_mini_cheetah.envs.mini_cheetah as e
import argparse
from fabulous.color import blue,green,red,bold
import numpy as np
import math
PI = np.pi



policy = np.load("experiments/19Dec1/iterations/best_policy.npy")

rpy_accurate = []
rpy_noisy = []
if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	parser.add_argument('--FrontMass', help='mass to be added in the first', type=float, default=0)
	parser.add_argument('--BackMass', help='mass to be added in the back', type=float, default=0)
	parser.add_argument('--FrictionCoeff', help='foot friction value to be set', type=float, default=0.6)
	parser.add_argument('--WedgeIncline', help='wedge incline degree of the wedge', type=int, default=11)
	parser.add_argument('--WedgeOrientation', help='wedge orientation degree of the wedge', type=float, default=0)
	parser.add_argument('--MotorStrength', help='maximum motor Strength to be applied', type=float, default=7.0)
	parser.add_argument('--RandomTest', help='flag to sample test values randomly ', type=bool, default=False)
	parser.add_argument('--seed', help='seed for the random sampling', type=float, default=100)
	parser.add_argument('--EpisodeLength', help='number of gait steps of a episode', type=int, default=1000)

	args = parser.parse_args()
	WedgePresent = False
	if(args.WedgeIncline == 0):
		WedgePresent = False
	
	env = e.MiniCheetahEnv(render=True, on_rack=True, gait = 'trot')
	steps = 0
	t_r = 0


	state = env.reset()


	#env._pybullet_client.resetDebugVisualizerCamera(1, -20, -20, [1, 0, 0])
	for i_step in range(args.EpisodeLength):

		action = policy.dot(state)
		state, r, _, angle = env.step(action)
		t_r += r

	print("Total_reward "+ str(t_r))
