# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

""" Run CPG """
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
foot_y = 0.0838  # this is the hip length
sideSign = np.array([-1, 1, -1, 1])  # get correct hip sign (body right is negative)
GAIT = "PACE"

env = QuadrupedGymEnv(render=True,  # visualize
                      on_rack=False,  # useful for debugging!
                      isRLGymInterface=False,  # not using RL
                      time_step=TIME_STEP,
                      action_repeat=1,
                      motor_control_mode="TORQUE",
                      add_noise=False,  # start in ideal conditions
                      # record_video=True
                      )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP, gait=GAIT)

TEST_STEPS = int(3 / (TIME_STEP))
t = np.arange(TEST_STEPS) * TIME_STEP

# [TODO] initialize data structures to save CPG and robot states
xyz_position_global = []
des_xyz_position_global = []
dxyz_position_global = []
joint_angles = []
des_joint_angles = []
r_dr_array = []
theta_dtheta_array = []
speed = np.array(np.empty((0, 3)))

############## Sample Gains
# joint PD gains
kp = np.array([100, 100, 100])
kd = np.array([2, 2, 2])
# Cartesian PD gains
kpCartesian = np.diag([500] * 3)
kdCartesian = np.diag([20] * 3)

for j in range(TEST_STEPS):
    # initialize torque array to send to motors
    action = np.zeros(12)
    # get desired foot positions from CPG
    xs, zs = cpg.update()
    r_dr_array.append(np.array([cpg.get_r(), cpg.get_dr()]))
    theta_dtheta_array.append(np.array([cpg.get_theta(), cpg.get_dtheta()]))

    # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
    q = env.robot.GetMotorAngles()
    dq = env.robot.GetMotorVelocities()

    # loop through desired foot positions and calculate torques
    xyz_pos = []
    des_xyz_pos = []
    dxyz_pos = []
    speed = np.append(speed, [env.get_speed()], axis=0)
    des_joint_angle_loop = np.empty((0, 3))
    joint_angles.append(np.reshape(q, (4, 3)))

    for i in range(4):
        q_leg = q[i * 3:(i + 1) * 3]
        dq_leg = dq[i * 3:(i + 1) * 3]
        # initialize torques for legi
        tau = np.zeros(3)
        # get desired foot i pos (xi, yi, zi) in leg frame
        leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])
        # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
        leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz)
        # Add joint PD contribution to tau for leg i (Equation 4)
        tau += kp * (leg_q - q_leg) + kd * (-dq_leg)  # [TODO]
        des_joint_angle_loop = np.append(des_joint_angle_loop, [leg_q], axis=0)

        # add Cartesian PD contribution
        if ADD_CARTESIAN_PD:
            # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
            J, pos = env.robot.ComputeJacobianAndPosition(i)
            # Get current foot velocity in leg frame (Equation 2)
            v = J @ dq_leg
            # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
            tau += J.T @ (kdCartesian @ (-v) + kpCartesian @ (leg_xyz - pos))  # [TODO]

            xyz_pos.append(pos)
            des_xyz_pos.append(leg_xyz)
            dxyz_pos.append(v)

        # Set tau for legi in action vector
        action[3 * i:3 * i + 3] = tau

    # send torques to robot and simulate TIME_STEP seconds
    env.step(action)

    # [TODO] save any CPG or robot states
    des_joint_angles.append(des_joint_angle_loop)
    xyz_position_global.append(xyz_pos)
    dxyz_position_global.append(dxyz_pos)
    des_xyz_position_global.append(des_xyz_pos)


#####################################################
# PLOTS
#####################################################
# example
def unpack(main_array: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return np.array([row[0] for row in main_array]), np.array([row[1] for row in main_array]), np.array(
        [row[2] for row in main_array]), np.array([row[3] for row in main_array])


def plot(fr: np.ndarray, fl: np.ndarray, rr: np.ndarray, rl: np.ndarray, t: np.ndarray, gait_name: str,
         indication: str = "position", des_fr: Union[np.ndarray, None] = None,
         des_fl: Union[np.ndarray, None] = None, des_rr: Union[np.ndarray, None] = None,
         des_rl: Union[np.ndarray, None] = None,
         labels: Tuple[str, str, str] = ("x", "y", "z")) -> None:
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))

    axes[0, 1].plot(t, fr[:, 0], label=labels[0], color='red')
    axes[0, 1].plot(t, fr[:, 1], label=labels[1], color='blue')
    axes[0, 1].plot(t, fr[:, 2], label=labels[2], color='magenta')
    axes[0, 1].set_title('FR ' + indication + ' ' + gait_name)

    axes[0, 0].plot(t, fl[:, 0], label=labels[0], color='red')
    axes[0, 0].plot(t, fl[:, 1], label=labels[1], color='blue')
    axes[0, 0].plot(t, fl[:, 2], label=labels[2], color='magenta')
    axes[0, 0].set_title('FL ' + indication + ' ' + gait_name)

    axes[1, 1].plot(t, rr[:, 0], label=labels[0], color='red')
    axes[1, 1].plot(t, rr[:, 1], label=labels[1], color='blue')
    axes[1, 1].plot(t, rr[:, 2], label=labels[2], color='magenta')
    axes[1, 1].set_title('RR ' + indication + ' ' + gait_name)

    axes[1, 0].plot(t, rl[:, 0], label=labels[0], color='red')
    axes[1, 0].plot(t, rl[:, 1], label=labels[1], color='blue')
    axes[1, 0].plot(t, rl[:, 2], label=labels[2], color='magenta')
    axes[1, 0].set_title('RL ' + indication + ' ' + gait_name)
    if des_fr is not None and des_fl is not None and des_rl is not None and des_rr is not None:
        axes[0, 1].plot(t, des_fr[:, 0], label="desired " + labels[0])
        axes[0, 1].plot(t, des_fr[:, 1], label="desired " + labels[1])
        axes[0, 1].plot(t, des_fr[:, 2], label="desired " + labels[2])

        axes[0, 0].plot(t, des_fl[:, 0], label="desired " + labels[0])
        axes[0, 0].plot(t, des_fl[:, 1], label="desired " + labels[1])
        axes[0, 0].plot(t, des_fl[:, 2], label="desired " + labels[2])

        axes[1, 1].plot(t, des_rr[:, 0], label="desired " + labels[0])
        axes[1, 1].plot(t, des_rr[:, 1], label="desired " + labels[1])
        axes[1, 1].plot(t, des_rr[:, 2], label="desired " + labels[2])

        axes[1, 0].plot(t, des_rl[:, 0], label="desired " + labels[0])
        axes[1, 0].plot(t, des_rl[:, 1], label="desired " + labels[1])
        axes[1, 0].plot(t, des_rl[:, 2], label="desired " + labels[2])

    axes[0, 0].legend()
    axes[1, 0].legend()
    axes[0, 1].legend()
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    del fig
    del axes


fr_xyz, fl_xyz, rr_xyz, rl_xyz = unpack(xyz_position_global)
fr_joint, fl_joint, rr_joint, rl_joint = unpack(joint_angles)
des_fr_joint, des_fl_joint, des_rr_joint, des_rl_joint = unpack(des_joint_angles)
fr_dxyz, fl_dxyz, rr_dxyz, rl_dxyz = unpack(dxyz_position_global)
des_fr_xyz, des_fl_xyz, des_rr_xyz, des_rl_xyz = unpack(des_xyz_position_global)

plot(fr_xyz, fl_xyz, rr_xyz, rl_xyz, t, GAIT, "position", des_fr_xyz, des_fl_xyz, des_rr_xyz, des_rl_xyz)
plot(fr_joint, fl_joint, rr_joint, rl_joint, t, GAIT, "angle", des_fr_joint, des_fl_joint, des_rr_joint,
     des_rl_joint, ("q0", "q1", "q2"))
plot(fr_dxyz, fl_dxyz, rr_dxyz, rl_dxyz, t, GAIT, "speed feet")

r_dr_array = np.array(r_dr_array)
plt.plot(t, r_dr_array[:, 0, 0], label="r0")
plt.plot(t, r_dr_array[:, 0, 1], label="r1")
plt.plot(t, r_dr_array[:, 0, 2], label="r2")
plt.plot(t, r_dr_array[:, 0, 3], label="r3")
plt.plot(t, r_dr_array[:, 1, 0], label="dr0")
plt.plot(t, r_dr_array[:, 1, 1], label="dr1")
plt.plot(t, r_dr_array[:, 1, 2], label="dr2")
plt.plot(t, r_dr_array[:, 1, 3], label="dr3")
plt.title("Plot of the rs and drs for gait " + GAIT)
plt.legend()
plt.show()

theta_dtheta_array = np.array(theta_dtheta_array)
plt.plot(t, theta_dtheta_array[:, 0, 0], label="theta0")
plt.plot(t, theta_dtheta_array[:, 0, 1], label="theta1")
plt.plot(t, theta_dtheta_array[:, 0, 2], label="theta2")
plt.plot(t, theta_dtheta_array[:, 0, 3], label="theta3")
plt.plot(t, theta_dtheta_array[:, 1, 0], label="dtheta0")
plt.plot(t, theta_dtheta_array[:, 1, 1], label="dtheta1")
plt.plot(t, theta_dtheta_array[:, 1, 2], label="dtheta2")
plt.plot(t, theta_dtheta_array[:, 1, 3], label="dtheta3")
plt.title("Plot of the theta and dthetas for gait " + GAIT)
plt.legend()
plt.show()

plt.plot(t, speed[:, 0], label='speed x', color='blue')
plt.plot(t, speed[:, 1], label='speed y', color='purple')
plt.plot(t, speed[:, 2], label='speed z', color='orange')
plt.title('speeds')
plt.legend()
plt.show()
