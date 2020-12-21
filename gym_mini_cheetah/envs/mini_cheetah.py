import numpy as np
import gym
from gym import spaces
import math
import random
from collections import deque
import pybullet
import gym_mini_cheetah.envs.bullet_client as bullet_client
import pybullet_data

LEG_POSITION = ["fl_", "bl_", "fr_", "br_"]
KNEE_CONSTRAINT_POINT_RIGHT = [0.014, 0, 0.076]  # hip
KNEE_CONSTRAINT_POINT_LEFT = [0.0, 0.0, -0.077]  # knee
RENDER_HEIGHT = 720
RENDER_WIDTH = 960
PI = np.pi
no_of_points = 100


def constrain_theta(theta):
    theta = np.fmod(theta, 2 * no_of_points)
    if (theta < 0):
        theta = theta + 2 * no_of_points
    return theta


class MiniCheetahEnv(gym.Env):

    def __init__(self,
				 render=False,
				 on_rack=False,
				 gait='trot',
				 phase=[0, no_of_points, no_of_points, 0],  # [FR, FL, BR, BL]
				 action_dim=4,
				 end_steps=1000,
				 stairs=False,
				 downhill=False,
				 seed_value=100,
				 wedge=True,
				 IMU_Noise=False,
				 deg=5):

        self._is_stairs = stairs
        self._is_wedge = wedge
        self._is_render = render
        self._on_rack = on_rack
        self.rh_along_normal = 0.24

        self.seed_value = seed_value
        random.seed(self.seed_value)

        if self._is_render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bullet_client.BulletClient()

        self._theta = 0

        self._frequency = -3
        self.termination_steps = end_steps
        self.downhill = downhill

        # PD gains
        self._kp = 500
        self._kd = 50

        self.dt = 0.005
        self._frame_skip = 25
        self._n_steps = 0
        self._action_dim = action_dim

        self._obs_dim = 8

        self.action = np.zeros(self._action_dim)

        self._last_base_position = [0, 0, 0]
        self.last_yaw = 0
        self._distance_limit = float("inf")

        self.current_com_height = 0.7


        if gait is 'trot':
            phase = [0, no_of_points, no_of_points, 0]
        elif gait is 'walk':
            phase = [0, no_of_points, 3 * no_of_points / 2, no_of_points / 2]
        self.inverse = False
        self._cam_dist = 1.0
        self._cam_yaw = 0.0
        self._cam_pitch = 0.0

        self.avg_vel_per_step = 0
        self.avg_omega_per_step = 0

        self.linearV = 0
        self.angV = 0
        self.friction = 0.7

        self.INIT_POSITION = [0, 0, 0.35]
        self.INIT_ORIENTATION = [0, 0, 0, 1]


        ## Gym env related mandatory variables
        observation_high = np.array([np.pi / 2] * self._obs_dim)
        observation_low = -observation_high
        self.observation_space = spaces.Box(observation_low, observation_high)

        action_high = np.array([1] * self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)

        self.hard_reset()


    def hard_reset(self):
        '''
		Function to
		1) Set simulation parameters which remains constant throughout the experiments
		2) load urdf of plane, wedge and robot in initial conditions
		'''
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(300))
        self._pybullet_client.setTimeStep(self.dt / self._frame_skip)

        self.plane = self._pybullet_client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
        self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
        self._pybullet_client.setGravity(0, 0, -9.8)


        self.MiniCheetah = self._pybullet_client.loadURDF("%s/mini_cheetah/mini_cheetah.urdf" % pybullet_data.getDataPath(), self.INIT_POSITION, self.INIT_ORIENTATION)

        self._joint_name_to_id, self._motor_id_list, self._motor_names = self.BuildMotorIdList()

        self.ResetLeg(reset_duration=1)

        if self._on_rack:
            self._pybullet_client.createConstraint(
                self.MiniCheetah, -1, -1, -1, self._pybullet_client.JOINT_FIXED,
                [0, 0, 0], [0, 0, 0], [0, 0, self.INIT_POSITION[2]])


        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])



    def reset(self):
        '''
		This function resets the environment
		Note : Set_Randomization() is called before reset() to either randomize or set environment in default conditions.
		'''
        self._theta = 0
        self._last_base_position = [0, 0, 0]
        self.last_yaw = 0
        self.inverse = False

        self._pybullet_client.resetBasePositionAndOrientation(self.MiniCheetah, self.INIT_POSITION, self.INIT_ORIENTATION)
        self._pybullet_client.resetBaseVelocity(self.MiniCheetah, [0, 0, 0], [0, 0, 0])
        self.ResetLeg()

        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
        self._n_steps = 0
        return self.GetObservation()



    def BuildMotorIdList(self):
        '''
		function to map joint_names with respective motor_ids as well as create a list of motor_ids
		Ret:
		joint_name_to_id : Dictionary of joint_name to motor_id
		motor_id_list	 : List of joint_ids for respective motors in order [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA ]
		'''
        self.num_joints = self._pybullet_client.getNumJoints(self.MiniCheetah)
        joint_name_to_id = {}
        for i in range(self.num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.MiniCheetah, i)
            joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

        # adding abduction
        MOTOR_NAMES = [#"torso_to_abduct_fr_j",
                       "abduct_fr_to_thigh_fr_j",
                       "thigh_fr_to_knee_fr_j",

                       #"torso_to_abduct_fl_j",
                       "abduct_fl_to_thigh_fl_j",
                       "thigh_fl_to_knee_fl_j",

                       #"torso_to_abduct_hr_j",
                       "abduct_hr_to_thigh_hr_j",
                       "thigh_hr_to_knee_hr_j",

                       #"torso_to_abduct_hl_j",
                       "abduct_hl_to_thigh_hl_j",
                       "thigh_hl_to_knee_hl_j"
                       ]


        motor_id_list = [joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]

        return joint_name_to_id, motor_id_list, MOTOR_NAMES

    def ResetLeg(self, reset_duration = 150):
        '''
		function to reset hip and knee joints' state
		Args:
			 leg_id 		  : denotes leg index
			 add_constraint   : bool to create constraints in lower joints of five bar leg mechanisim
			 standstilltorque : value of initial torque to set in hip and knee motors for standing condition
		'''
        for i in range(8):
            if i%2 == 0:
                joint_reset_angle = -1*math.radians(50)
            elif i%2 ==1:
                joint_reset_angle = math.radians(110)


            self._pybullet_client.resetJointState(
                self.MiniCheetah,
                self._joint_name_to_id[self._motor_names[i]],  # motor
                targetValue=joint_reset_angle, targetVelocity=0)

        for t in range(reset_duration):
            if t < reset_duration - 1:
                standing_motor_force = 17
            else:
                standing_motor_force = 0
            for id in self._motor_id_list:
                self._pybullet_client.setJointMotorControl2(
                    bodyIndex=self.MiniCheetah,
                    jointIndex=id,
                    controlMode=self._pybullet_client.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=standing_motor_force
                )
                self._pybullet_client.stepSimulation()



    def SetMotorTorqueById(self, motor_id, torque):
        '''
        function to set motor torque for respective motor_id
        '''
        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.MiniCheetah,
            jointIndex=motor_id,
            controlMode=self._pybullet_client.TORQUE_CONTROL,
            force=torque)

    def SetMotorPositionById(self, motor_id, position):
        '''
        function to set motor torque for respective motor_id
        '''
        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.MiniCheetah,
            jointIndex=motor_id,
            controlMode=self._pybullet_client.POSITION_CONTROL,
            targetPosition= position,
            force = 17,
            positionGain = 300,
            velocityGain = 20
        )
    def _apply_pd_control(self, motor_commands, motor_vel_commands):
        '''
        Apply PD control to reach desired motor position commands
        Ret:
            applied_motor_torque : array of applied motor torque values in order [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA]
        '''
        self._kp = 300
        self._kd = 20
        qpos_act = self.GetMotorAngles()
        qvel_act = self.GetMotorVelocities()
        applied_motor_torque = self._kp * (motor_commands - qpos_act) + self._kd * (motor_vel_commands - qvel_act)

        motor_strength = 15
        applied_motor_torque = np.clip(np.array(applied_motor_torque), -motor_strength, motor_strength)
        applied_motor_torque = applied_motor_torque.tolist()

        for motor_id, motor_torque in zip(self._motor_id_list, applied_motor_torque):
            self.SetMotorTorqueById(motor_id, motor_torque)
        return applied_motor_torque
    def GetObservation(self):
        '''
        This function returns the current observation of the environment for the interested task
        Ret:
            obs : [R(t-2), P(t-2), Y(t-2), R(t-1), P(t-1), Y(t-1), R(t), P(t), Y(t), estimated support plane (roll, pitch) ]
        '''
        pos, ori = self.GetBasePosAndOrientation()
        motor_angles = self.GetMotorAngles()
        RPY = self._pybullet_client.getEulerFromQuaternion(ori)
        RPY = np.round(RPY, 5)
        ang_vel = self.GetBaseAngularVelocity()
        lin_vel = self.GetBaseLinearVelocity()
        obs = np.concatenate(([pos[2]], lin_vel,[ang_vel[0], ang_vel[1]], [RPY[0], RPY[1]])).ravel()

        return obs


    def GetMotorAngles(self):
        '''
		This function returns the current joint angles in order [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA ]
		'''
        motor_ang = [self._pybullet_client.getJointState(self.MiniCheetah, motor_id)[0] for motor_id in self._motor_id_list]
        return motor_ang

    def GetMotorVelocities(self):
        '''
        This function returns the current joint velocities in order [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA ]
        '''
        motor_vel = [self._pybullet_client.getJointState(self.MiniCheetah, motor_id)[1] for motor_id in
                     self._motor_id_list]
        return motor_vel

    def GetBasePosAndOrientation(self):
        '''
        This function returns the robot torso position(X,Y,Z) and orientation(Quaternions) in world frame
        '''
        position, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.MiniCheetah))
        return position, orientation

    def GetBaseAngularVelocity(self):
        '''
        This function returns the robot base angular velocity in world frame
        Ret: list of 3 floats
        '''
        basevelocity = self._pybullet_client.getBaseVelocity(self.MiniCheetah)
        return basevelocity[1]

    def GetBaseLinearVelocity(self):
        '''
        This function returns the robot base linear velocity in world frame
        Ret: list of 3 floats
        '''
        basevelocity = self._pybullet_client.getBaseVelocity(self.MiniCheetah)
        return basevelocity[0]

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
        action = self.transform_action(action)

        self.do_simulation(action, n_frames=self._frame_skip)

        ob = self.GetObservation()
        reward, done = self._get_reward()
        return ob, reward, done, {}

    def transform_action(self, action):
        action = np.clip(action, -1, 1)
        action[0] = action[0]*math.radians(35) - math.radians(50)
        action[2] = action[2] * math.radians(35) - math.radians(50)
        action[1] = (action[1] + 1)/2 * math.radians(100) + math.radians(50)
        action[3] = (action[3] +1)/2* math.radians(100) + math.radians(50)
        aug_action = np.array([action[0], action[1], action[2], action[3],action[2], action[3], action[0], action[1]])
        return aug_action

    def do_simulation(self, action, n_frames):
        abd_motor_ids = [0,4,8,12]
        for _ in range(n_frames):
            _ = self._apply_pd_control(action, np.zeros(8))
            # for motor_id, motor_angle in zip(self._motor_id_list,action):
            #     self.SetMotorPositionById(motor_id, motor_angle)
            for abd_id in abd_motor_ids:
                self.SetMotorPositionById(abd_id,0)
            self._pybullet_client.stepSimulation()
        self._n_steps += 1

    def _termination(self, pos, orientation):
        '''
		Check termination conditions of the environment
		Args:
			pos 		: current position of the robot's base in world frame
			orientation : current orientation of robot's base (Quaternions) in world frame
		Ret:
			done 		: return True if termination conditions satisfied
		'''
        done = False
        RPY = self._pybullet_client.getEulerFromQuaternion(orientation)

        if self._n_steps >= self.termination_steps:
            done = True
        else:
            if abs(RPY[0]) > math.radians(60):
                print('Oops, Robot about to fall sideways! Terminated')
                done = True

            if abs(RPY[1]) > math.radians(40):
                print('Oops, Robot doing wheely! Terminated')
                done = True

            if pos[2] > 0.5 :
                print('Robot was too high! Terminated')
                done = True
            if pos[2] < 0.08 :
                print('Robot was too low! Terminated')
                done = True

        return done

    def _get_reward(self):
        '''
        Calculates reward achieved by the robot for RPY stability, torso height criterion and forward distance moved on the slope:
        Ret:
            reward : reward achieved
            done   : return True if environment terminates

        '''

        pos, ori = self.GetBasePosAndOrientation()

        RPY_orig = self._pybullet_client.getEulerFromQuaternion(ori)
        RPY = np.round(RPY_orig, 4)

        current_height = round(pos[2], 5)
        desired_height = 0.24

        roll_reward = np.exp(-20 * ((RPY[0] ) ** 2))
        pitch_reward = np.exp(-35 * ((RPY[1]) ** 2))
        yaw_reward = np.exp(-30 * (RPY[2] ** 2))
        height_reward = np.exp(-350 * (desired_height - current_height) ** 2)

        x = pos[0]
        x_l = self._last_base_position[0]
        self._last_base_position = pos

        step_distance_x = (x - x_l)
        penalty = 0

        if abs(step_distance_x) <= 0.00003:
            penalty = 1


        done = self._termination(pos, ori)
        if done:
            reward = 0
        else:
            reward = round(pitch_reward, 4) + round(roll_reward, 4) + round(height_reward, 4) + \
                     150 * round(step_distance_x, 4) - penalty*0.5
        # print(pitch_reward, roll_reward,height_reward)

        return reward, done


# env = MiniCheetahEnv(render=True, on_rack=False)
# env.reset()
# for _ in range(1000):
#     action =  np.ones(4)*0
#     env.step(action)


