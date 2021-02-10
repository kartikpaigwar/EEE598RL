import gym
from gym import spaces
import numpy as np
import math
import pybullet


class MiniCheetahEnv1(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 render=False,
                 on_rack=False,
                 end_steps=800,
                 save_path=None
                 ):

        import gym_mini_cheetah.envs.mini_cheetah as robot
        self.mini_cheetah = robot.MiniCheetah(render=render, on_rack=on_rack,end_steps=end_steps, video_path=save_path)

        self._action_dim = 8

        self._obs_dim = 18

        self.action = np.zeros(self._action_dim)
        motor_space_low = [-np.pi/3, np.pi/4, -np.pi/3, np.pi/4, -np.pi/3, np.pi/4, -np.pi/3, np.pi/4]
        motor_space_high = [0, math.radians(105),0,math.radians(105),0,math.radians(105),0,math.radians(105)]
        observation_low = np.array(motor_space_low + [0, 0, -1, -0.5, -np.pi / 2, -np.pi/2, -np.pi/2, -np.pi / 2, -np.pi / 2, -np.pi / 2])
        observation_high = np.array(motor_space_high + [0.5, 3, 1, 0.5, np.pi / 2, np.pi / 2, np.pi/2, np.pi/2, np.pi / 2, np.pi / 2])

        self.observation_space = spaces.Box(observation_low, observation_high)

        action_high = np.array([1] * self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)



    def reset(self):
        """
        This function resets the environment
        """
        self.mini_cheetah.reset_robot()
        return self.GetObservation()


    def GetObservation(self):
        """
        This function returns the current observation of the environment for the interested task.
        Obs_Dimension = 8 + 10

        :return: [hip_knee_angles, robot_z_pos, lin_vel_x, lin_vel_y, lin_vel_z, roll rate, pitch rate, roll, pitch]
        """

        pos, ori = self.mini_cheetah.GetBasePosAndOrientation()
        motor_angles = self.mini_cheetah.GetMotorAngles()
        hip_knee_angles = []
        for i in range(0, len(motor_angles)):
            if i % 3 > 0:
                hip_knee_angles.append(round(motor_angles[i], 6))
        RPY = pybullet.getEulerFromQuaternion(ori)
        RPY = np.round(RPY, 4)
        ang_vel = self.mini_cheetah.GetBaseAngularVelocity()
        lin_vel = self.mini_cheetah.GetBaseLinearVelocity()
        obs = np.concatenate((hip_knee_angles, [pos[2]], lin_vel, [ang_vel[0], ang_vel[1], ang_vel[2]], [RPY[0], RPY[1], RPY[2]])).ravel()

        return obs


    def GetObservation_Orig(self):
        """
        This function returns the current observation of the environment for the interested task.
        Obs_Dimension = 8

        :return: [robot_z_pos, lin_vel_x, lin_vel_y, lin_vel_z, roll rate, pitch rate, roll, pitch]
        """

        pos, ori = self.mini_cheetah.GetBasePosAndOrientation()
        motor_angles = self.mini_cheetah.GetMotorAngles()
        RPY = pybullet.getEulerFromQuaternion(ori)
        RPY = np.round(RPY, 4)
        ang_vel = self.mini_cheetah.GetBaseAngularVelocity()
        lin_vel = self.mini_cheetah.GetBaseLinearVelocity()
        obs = np.concatenate(([pos[2]], lin_vel, [ang_vel[0], ang_vel[1], ang_vel[2]], [RPY[0], RPY[1], RPY[2]])).ravel()

        return obs


    def step(self, action):
        """
        function to perform one step in the environment
        :param action: 4 dimension array of action values ranging from [-1,1]
        :return:
        ob 	   : observation after taking step
        reward     : reward received after taking step
        done       : whether the step terminates the env
        {}	   : any information of the env (will be added later)
        """

        self.transform_action_2_motor_commands(action)

        self.mini_cheetah.do_simulation()

        ob = self.GetObservation()
        reward, done = self._get_reward()
        return ob, reward, done, {}

    def transform_action_2_motor_commands(self, action):
        """
        Transform normalized actions and map to corresponding motor angles of each leg
        :param action: 4 dimension array of action values ranging from [-1,1]
        :return: None
        """
        action = np.clip(action, -1, 1)
        action[:4] = action[:4] * math.radians(30) - math.radians(30)
        action[4:] = (action[4:] + 1) / 2 * math.radians(60) + math.radians(45)

        for leg in self.mini_cheetah.legs:
            leg.abduction_motor_angle = 0
            if leg.name == "fr":
                leg.hip_motor_angle = action[0]
                leg.knee_motor_angle = action[4]
            if leg.name == "fl":
                leg.hip_motor_angle = action[1]
                leg.knee_motor_angle = action[5]
            if leg.name == "br":
                leg.hip_motor_angle = action[2]
                leg.knee_motor_angle = action[6]
            if leg.name == "bl":
                leg.hip_motor_angle = action[3]
                leg.knee_motor_angle = action[7]


    def _get_reward(self):
        """
        Calculates reward achieved by the robot for Roll Pitch stability, torso height criterion and forward distance moved :
        :return:
        reward : reward achieved
        done   : return True if environment terminates
        """

        pos, ori = self.mini_cheetah.GetBasePosAndOrientation()
        base_vel = self.mini_cheetah.GetBaseLinearVelocity()
        RPY_orig = pybullet.getEulerFromQuaternion(ori)
        RPY = np.round(RPY_orig, 4)

        current_height = round(pos[2], 5)

        desired_height = 0.27
        desired_vel = 0.6

        roll_reward = np.exp(-27 * ((RPY[0]) ** 2)) #20
        pitch_reward = np.exp(-50 * ((RPY[1]) ** 2))   #35
        yaw_reward = np.exp(-20 * ((RPY[2]) ** 2))   #35
        height_reward = np.exp(-900 * (desired_height - current_height) ** 2)  #350
        zvel_reward = 0 #np.exp(-1.5*(base_vel[2]**2))
        xvel_reward = np.exp(-9 * ((desired_vel - base_vel[0]) ** 2))
        #Calculate distance moved along x direction from its last position
        x = pos[0]
        x_l = self.mini_cheetah._last_base_position[0]
        self.mini_cheetah._last_base_position = pos
        step_distance_x = (x - x_l)
        step_distance_x_reward = np.clip(320*step_distance_x,-1,1) #clip reward between [-1,1]

        # Penalize if the robot remains standstill
        penalty = 0
        if step_distance_x <= 0.00007:
            penalty = 3

        # Check if episode terminates
        done, system_penalty = self.mini_cheetah._termination()
        if done:
            reward = 0
        else:
            reward = round(pitch_reward, 4) + round(roll_reward, 4) + round(height_reward, 4) + \
                     round(yaw_reward, 4) + round(xvel_reward, 4) + round(zvel_reward, 4) + step_distance_x_reward - penalty - system_penalty
        #print("Xvel", height_reward, current_height, xvel_reward, base_vel[0], step_distance_x_reward)
        return reward, done

    def render(self, mode="rgb_array", close=False):
        render_array = self.mini_cheetah.render(mode,close)
        return render_array
