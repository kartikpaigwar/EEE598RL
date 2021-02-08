import numpy as np
import math
import pybullet
from dataclasses import dataclass
from collections import namedtuple
import utils.bullet_client as bullet_client
import pybullet_data
import os

@dataclass
class leg_data:
    name: str
    abduction_motor_id: int = 0
    hip_motor_id: int = 0
    knee_motor_id: int = 0
    toe_joint_id: int = 0
    abduction_motor_angle: float = 0.0
    hip_motor_angle: float = 0.0
    knee_motor_angle: float = 0.0

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class MiniCheetah():
    """The MiniCheetah class that simulates a quadruped robot from MiT.
    """
    def __init__(self,
                 render=False,
                 on_rack=False,
                 end_steps=800,
                 default=False
                 ):

        self.default_urdf = default

        self._is_render = render
        self._on_rack = on_rack

        if self._is_render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bullet_client.BulletClient()

        self.termination_steps = end_steps

        self.dt = 0.005               #Control Time Step
        self._frame_skip = 25         #Frames to Repeat Action
        self._n_steps = 0

        # Reset position and orientation of the robot
        self.INIT_POSITION = [0, 0, 0.35]
        self.INIT_ORIENTATION = [0, 0, 0, 1]
        self._last_base_position = [0, 0, 0]


        self._kp = 300                 #Position Gain
        self._kd = 20                  #Velocity Gain
        self.motor_strength = 15       #Peak Torque

        #Rendering Camera Parameters
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40

        #Instantiate dataclass for each leg [FR, FL, BR, BL]
        self.front_left = leg_data('fl')
        self.front_right = leg_data('fr')
        self.back_left = leg_data('bl')
        self.back_right = leg_data('br')

        Legs = namedtuple('legs', 'front_right front_left back_right back_left')
        self.legs = Legs(front_right=self.front_right, front_left=self.front_left, back_right=self.back_right,
                    back_left=self.back_left)

        self.hard_reset()

    def hard_reset(self):
        """
        Function to
        1) Set simulation parameters which remains constant throughout the experiments
        2) load urdf of plane and robot in initial conditions
        3) build motor ids list, joint name to ids dictionary
        :return:
        """
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(300))
        self._pybullet_client.setTimeStep(self.dt / self._frame_skip)

        self.plane = self._pybullet_client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
        self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
        self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=1.0)

        self._pybullet_client.setGravity(0, 0, -9.8)

        if self.default_urdf:
            self.MiniCheetah = self._pybullet_client.loadURDF(
                "%s/mini_cheetah/mini_cheetah.urdf" % pybullet_data.getDataPath(), self.INIT_POSITION,
                self.INIT_ORIENTATION)
        else:
            robot_urdf_path = os.path.realpath("../../urdf/mini_cheetah_simple.urdf")
            self.MiniCheetah = self._pybullet_client.loadURDF(robot_urdf_path, self.INIT_POSITION,
                                                              self.INIT_ORIENTATION)

        self._joint_name_to_id, self._motor_id_list = self.BuildMotorIdList()

        self.ResetLegs(reset_duration=1)

        if self._on_rack:
            self._pybullet_client.createConstraint(
                self.MiniCheetah, -1, -1, -1, self._pybullet_client.JOINT_FIXED,
                [0, 0, 0], [0, 0, 0], [0, 0, self.INIT_POSITION[2]])

        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])

    def reset_robot(self):
        """
        Reset robot in standing posture at initial position and orientation
        """

        self._last_base_position = [0, 0, 0]
        self._pybullet_client.resetBasePositionAndOrientation(self.MiniCheetah, self.INIT_POSITION, self.INIT_ORIENTATION)
        self._pybullet_client.resetBaseVelocity(self.MiniCheetah, [0, 0, 0], [0, 0, 0])
        self.ResetLegs()
        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
        self._n_steps = 0

    def BuildMotorIdList(self):
        """
        function to map joint_names with respective motor_ids as well as create a list of motor_ids
        Ret:
        joint_name_to_id : Dictionary of joint_name to motor_id
        motor_id_list	 : List of joint_ids for respective motors in order [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA ]
        """
        self.num_joints = self._pybullet_client.getNumJoints(self.MiniCheetah)
        joint_name_to_id = {}

        if self.default_urdf:
            toe_joint = ["toe_fr_joint", "toe_fl_joint", "toe_hr_joint", "toe_hl_joint"]
        else:
            toe_joint = ["fixed_shank_to_foot_fr", "fixed_shank_to_foot", "fixed_shank_to_foot_hr",
                         "fixed_shank_to_foot_hl"]

        for i in range(self.num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.MiniCheetah, i)
            joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

        self.front_right.abduction_motor_id = joint_name_to_id["torso_to_abduct_fr_j"]
        self.front_right.hip_motor_id = joint_name_to_id["abduct_fr_to_thigh_fr_j"]
        self.front_right.knee_motor_id = joint_name_to_id["thigh_fr_to_knee_fr_j"]
        self.front_right.toe_joint_id = joint_name_to_id[toe_joint[0]]

        self.front_left.abduction_motor_id = joint_name_to_id["torso_to_abduct_fl_j"]
        self.front_left.hip_motor_id = joint_name_to_id["abduct_fl_to_thigh_fl_j"]
        self.front_left.knee_motor_id = joint_name_to_id["thigh_fl_to_knee_fl_j"]
        self.front_left.toe_joint_id = joint_name_to_id[toe_joint[1]]

        self.back_right.abduction_motor_id = joint_name_to_id["torso_to_abduct_hr_j"]
        self.back_right.hip_motor_id = joint_name_to_id["abduct_hr_to_thigh_hr_j"]
        self.back_right.knee_motor_id = joint_name_to_id["thigh_hr_to_knee_hr_j"]
        self.back_right.toe_joint_id = joint_name_to_id[toe_joint[2]]

        self.back_left.abduction_motor_id = joint_name_to_id["torso_to_abduct_hl_j"]
        self.back_left.hip_motor_id = joint_name_to_id["abduct_hl_to_thigh_hl_j"]
        self.back_left.knee_motor_id = joint_name_to_id["thigh_hl_to_knee_hl_j"]
        self.back_left.toe_joint_id = joint_name_to_id[toe_joint[3]]

        motor_id_list = []
        for leg in self.legs:
            motor_id_list.append(leg.abduction_motor_id)
            motor_id_list.append(leg.hip_motor_id)
            motor_id_list.append(leg.knee_motor_id)

        return joint_name_to_id, motor_id_list

    def ResetLegs(self, reset_duration=200):
        """
        function to reset abduction, hip and knee joints' in standing posture
        Args:
             reset_duration   : simulation time steps until robot remain in standing posture

        """
        for leg in self.legs:
            # Precalculated joints angle of each leg in standing position
            leg.abduction_motor_angle = 0
            leg.hip_motor_angle = -1*math.radians(50)
            leg.knee_motor_angle = math.radians(110)

            self._pybullet_client.resetJointState(
                self.MiniCheetah,
                leg.abduction_motor_id,
                targetValue=leg.abduction_motor_angle, targetVelocity=0)
            self._pybullet_client.resetJointState(
                self.MiniCheetah,
                leg.hip_motor_id,
                targetValue=leg.hip_motor_angle, targetVelocity=0)
            self._pybullet_client.resetJointState(
                self.MiniCheetah,
                leg.knee_motor_id,
                targetValue=leg.knee_motor_angle, targetVelocity=0)

        for t in range(reset_duration):

            if t < reset_duration - 1:
                #Apply constant motor force to remain in standing position
                standing_motor_force = 17
            else:
                #Free joints once reset duration is over
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
        """
        function to set motor torque for respective motor_id
        :param motor_id: int
        :param torque: float
        """

        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.MiniCheetah,
            jointIndex=motor_id,
            controlMode=self._pybullet_client.TORQUE_CONTROL,
            force=torque)

    def SetMotorPositionById(self, motor_id, position):
        """
        function to set motor position for respective motor_id
        :param motor_id: int
        :param position: radians
        """
        self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.MiniCheetah,
            jointIndex=motor_id,
            controlMode=self._pybullet_client.POSITION_CONTROL,
            targetPosition=position,
            force=self.motor_strength,
            positionGain=self._kp,
            velocityGain=self._kd
        )

    def _apply_pd_control(self, motor_commands, motor_vel_commands):
        """
        Apply PD control to reach desired motor position commands and velocity commands

        Ret:
        applied_motor_torque : array of applied motor torque values in order of motor_ids
        """

        qpos_act = self.GetMotorAngles()
        qvel_act = self.GetMotorVelocities()

        applied_motor_torque = self._kp * (motor_commands - qpos_act) + self._kd * (motor_vel_commands - qvel_act)

        applied_motor_torque = np.clip(np.array(applied_motor_torque), -self.motor_strength, self.motor_strength)
        applied_motor_torque = applied_motor_torque.tolist()

        for motor_id, motor_torque in zip(self._motor_id_list, applied_motor_torque):
            self.SetMotorTorqueById(motor_id, motor_torque)
        return applied_motor_torque


    def GetMotorAngles(self):
        """
        This function returns the current joint angles in order of  motor ids
        """
        motor_ang = [self._pybullet_client.getJointState(self.MiniCheetah, motor_id)[0] for motor_id in
                     self._motor_id_list]

        return motor_ang

    def GetMotorVelocities(self):
        """
        This function returns the current joint velocities in order of motor ids
        """
        motor_vel = [self._pybullet_client.getJointState(self.MiniCheetah, motor_id)[1] for motor_id in
                     self._motor_id_list]
        return motor_vel

    def GetBasePosAndOrientation(self):
        """
        This function returns the robot torso position(X,Y,Z) and orientation(Quaternions) in world frame
        """
        position, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.MiniCheetah))
        return position, orientation

    def GetBaseAngularVelocity(self):
        """
        This function returns the robot base angular velocity in world frame
        Ret: list of 3 floats
        """
        basevelocity = self._pybullet_client.getBaseVelocity(self.MiniCheetah)
        return basevelocity[1]

    def GetBaseLinearVelocity(self):
        """
        This function returns the robot base linear velocity in world frame
        :return: list of 3 floats
        """
        basevelocity = self._pybullet_client.getBaseVelocity(self.MiniCheetah)
        return basevelocity[0]

    def GetMotorCommands(self):
        """
        This is function used to retrieve motor_angles assigned in each of the leg dataclasses
        :return:
        """
        motor_commands = []
        for leg in self.legs:
            motor_commands.append(leg.abduction_motor_angle)
            motor_commands.append(leg.hip_motor_angle)
            motor_commands.append(leg.knee_motor_angle)
        return  np.array(motor_commands)

    def SetMotorCommands(self, motor_ang):
        """
        This is function used to assign motor_angles in each of the leg dataclasses
        :param motor_ang: list of 12 motor angles in order of motor ids
        :return: None
        """
        leg_id = 0
        for leg in self.legs:
            leg.abduction_motor_angle = motor_ang[leg_id]
            leg.hip_motor_angle = motor_ang[leg_id + 1]
            leg.knee_motor_angle = motor_ang[leg_id + 2]
            leg_id += 1

    def do_simulation(self):
        """
        Function to simulate motor commands
        :return: list of 12 torque values applied in each of the motors
        """
        motor_commands = self.GetMotorCommands()
        velocity_commands = np.zeros(12)
        self.shank_contacts = np.zeros(4)
        for _ in range(self._frame_skip):
            applied_motor_torques = self._apply_pd_control(motor_commands, velocity_commands)
            self.shank_contacts = self.check_shank_contact()

            self._pybullet_client.stepSimulation()
        self._n_steps += 1

    def _termination(self):
        """
        Check termination conditions of the environment
        Args:
            pos 		: current position of the robot's base in world frame
            orientation : current orientation of robot's base (Quaternions) in world frame
        Ret:
            done 		: return True if termination conditions satisfied
        """

        done = False
        debug = False    # Debug with True for better intuition of the performance during the training
        penalty = 0

        pos, orientation = self.GetBasePosAndOrientation()
        RPY = self._pybullet_client.getEulerFromQuaternion(orientation)

        if self._n_steps >= self.termination_steps:
            done = True
        else:
            if abs(RPY[0]) > math.radians(60):
                if debug:
                    print('Oops, Robot about to fall sideways! Terminated')
                done = True

            if abs(RPY[1]) > math.radians(50):
                if debug:
                    print('Oops, Robot doing wheely! Terminated')
                done = True

            if pos[2] > 0.5:
                if debug:
                    print('Robot was too high! Terminated')
                done = True
            if 0.37 < pos[2] < 0.5:
                penalty += 0.5

            if pos[2] < 0.10:
                if debug:
                    print('Robot was too low! Terminated')
                done = True
            if 0.1 < pos[2] < 0.24:
                penalty += 0.3

            penalty = penalty + sum(self.shank_contacts)/4.0
            if debug:
                print("Shank_touch penalty : ", penalty)

        return done, penalty

    def check_shank_contact(self):
        '''
        Retrieve foot contact information with the supporting ground and any special structure (wedge/stairs).
        Ret:
            foot_contact_info : 8 dimensional binary array, first four values denote contact information of feet [FR, FL, BR, BL] with the ground
            while next four with the special structure.
        '''
        shank_ids = [2,6,10,14]
        contact_dist = self._pybullet_client.getLinkStates(self.MiniCheetah, shank_ids)

        for idx in range(4):
            if contact_dist[idx][4][2] < 0.03:
                self.shank_contacts[idx] = 1

        return self.shank_contacts


    def render(self, mode="rgb_array", close=False):
        """
        Need to verify. Currently, connecting GUI for rendering
        """
        if mode != "rgb_array":
            return np.array([])
        base_pos, orn = self._pybullet_client.getBasePositionAndOrientation(self.MiniCheetah)
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                              distance=self._cam_dist,
                                                                              yaw=self._cam_yaw,
                                                                              pitch=self._cam_pitch,
                                                                              roll=0,
                                                                              upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                       aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                                       nearVal=0.1,
                                                                       farVal=100.0)
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(width=RENDER_WIDTH,
                                                                height=RENDER_HEIGHT,
                                                                viewMatrix=view_matrix,
                                                                projectionMatrix=proj_matrix,
                                                                renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

