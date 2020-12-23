import gym
from gym import spaces
import numpy as np
import math
import pybullet


class MiniCheetahEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 render=False,
                 on_rack=False,
                 end_steps=800
                 ):

        import gym_mini_cheetah.envs.mini_cheetah as robot
        self.mini_cheetah = robot.MiniCheetah(render=render, on_rack=on_rack,end_steps=end_steps)

        self._action_dim = 4

        self._obs_dim = 8

        self.action = np.zeros(self._action_dim)

        observation_low = np.array([0, 0, -1, -0.5, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2])
        observation_high = np.array([0.5, 3, 1, 0.5, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2])
        self.observation_space = spaces.Box(observation_low, observation_high)

        action_high = np.array([1] * self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)



    def reset(self):
        '''
    This function resets the environment
    '''
        self.mini_cheetah.reset()
        return self.GetObservation()


    def GetObservation(self):
        '''
    This function returns the current observation of the environment for the interested task
    Ret:
        obs : [R(t-2), P(t-2), Y(t-2), R(t-1), P(t-1), Y(t-1), R(t), P(t), Y(t), estimated support plane (roll, pitch) ]
    '''
        pos, ori = self.mini_cheetah.GetBasePosAndOrientation()
        motor_angles = self.mini_cheetah.GetMotorAngles()
        RPY = pybullet.getEulerFromQuaternion(ori)
        RPY = np.round(RPY, 4)
        ang_vel = self.mini_cheetah.GetBaseAngularVelocity()
        lin_vel = self.mini_cheetah.GetBaseLinearVelocity()
        obs = np.concatenate(([pos[2]], lin_vel, [ang_vel[0], ang_vel[1]], [RPY[0], RPY[1]])).ravel()

        return obs

    def step(self, action):
        '''
    function to perform one step in the environment
    Args:
        action : array of action values
    Ret:
        ob 	   : observation after taking step
        reward     : reward received after taking step
        done       : whether the step terminates the env
        {}	   : any information of the env (will be added later)
    '''
        self.transform_action_2_motor_commands(action)

        self.mini_cheetah.do_simulation()

        ob = self.GetObservation()
        reward, done = self._get_reward()
        return ob, reward, done, {}

    def transform_action_2_motor_commands(self, action):
        action = np.clip(action, -1, 1)
        action[0] = action[0] * math.radians(40) - math.radians(50)
        action[2] = action[2] * math.radians(40) - math.radians(50)
        action[1] = (action[1] + 1) / 2 * math.radians(80) + math.radians(50)
        action[3] = (action[3] + 1) / 2 * math.radians(80) + math.radians(50)

        for leg in self.mini_cheetah.legs:
            leg.abduction_motor_angle = 0
            if leg.name=="fr" or leg.name=="bl":
                leg.hip_motor_angle = action[0]
                leg.knee_motor_angle = action[1]
            else:
                leg.hip_motor_angle = action[2]
                leg.knee_motor_angle = action[3]


    def _get_reward(self):
        '''
    Calculates reward achieved by the robot for RPY stability, torso height criterion and forward distance moved on the slope:
    Ret:
        reward : reward achieved
        done   : return True if environment terminates

    '''

        pos, ori = self.mini_cheetah.GetBasePosAndOrientation()

        RPY_orig = pybullet.getEulerFromQuaternion(ori)
        RPY = np.round(RPY_orig, 4)

        current_height = round(pos[2], 5)
        desired_height = 0.24

        roll_reward = np.exp(-20 * ((RPY[0]) ** 2))
        pitch_reward = np.exp(-35 * ((RPY[1]) ** 2))
        yaw_reward = np.exp(-30 * (RPY[2] ** 2))
        height_reward = np.exp(-350 * (desired_height - current_height) ** 2)

        x = pos[0]
        x_l = self.mini_cheetah._last_base_position[0]
        self.mini_cheetah._last_base_position = pos

        step_distance_x = (x - x_l)
        penalty = 0

        if abs(step_distance_x) <= 0.00003:
            penalty = 0.5
        step_distance_x_reward = np.clip(200*step_distance_x,-1,1)
        done = self.mini_cheetah._termination()
        if done:
            reward = 0
        else:
            reward = round(pitch_reward, 4) + round(roll_reward, 4) + round(height_reward, 4) + \
                     step_distance_x_reward - penalty
        # print(pitch_reward, roll_reward,height_reward, step_distance_x_reward)

        return reward, done

    def render(self, mode="rgb_array", close=False):
        render_array = self.mini_cheetah.render(mode,close)
        return render_array
