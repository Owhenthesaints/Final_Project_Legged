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
cpg = HopfNetwork(time_step=TIME_STEP, gait="PACE")

TEST_STEPS = int(3 / (TIME_STEP))
t = np.arange(TEST_STEPS) * TIME_STEP

# [TODO] initialize data structures to save CPG and robot states
xyz_position_global = []
joint_angles = []

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
    # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
    q = env.robot.GetMotorAngles()
    dq = env.robot.GetMotorVelocities()

    # loop through desired foot positions and calculate torques
    xyz_pos = []
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

        # add Cartesian PD contribution
        if ADD_CARTESIAN_PD:
            # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
            # [TODO]
            J, pos = env.robot.ComputeJacobianAndPosition(i)
            # Get current foot velocity in leg frame (Equation 2)
            # [TODO]
            v = J @ dq_leg  # to check if it is right
            # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
            tau += J.T @ (kdCartesian @ (-v) + kpCartesian @ (leg_xyz - pos))  # [TODO]

            xyz_pos.append(pos)

        # Set tau for legi in action vector
        action[3 * i:3 * i + 3] = tau

    # send torques to robot and simulate TIME_STEP seconds
    env.step(action)

    # [TODO] save any CPG or robot states
    xyz_position_global.append(xyz_pos)

#####################################################
# PLOTS
#####################################################
# example
fr_xyz = np.array([row[0] for row in xyz_position_global])
fl_xyz = np.array([row[1] for row in xyz_position_global])
rr_xyz = np.array([row[2] for row in xyz_position_global])
rl_xyz = np.array([row[3] for row in xyz_position_global])
fr_joint = np.array([row[0] for row in joint_angles])
fl_joint = np.array([row[1] for row in joint_angles])
rr_joint = np.array([row[2] for row in joint_angles])
rl_joint = np.array([row[3] for row in joint_angles])
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))

axes[0, 1].plot(t, fr_xyz[:, 1], label='FR y', color='blue')
axes[0, 1].plot(t, fr_xyz[:, 0], label='FR x', color='red')
axes[0, 1].plot(t, fr_xyz[:, 2], label='FR z', color='magenta')
axes[0, 1].set_title('FR position ' + cpg.GAIT)
axes[0, 1].legend()

axes[0, 0].plot(t, fl_xyz[:, 0], label='FL x', color='red')
axes[0, 0].plot(t, fl_xyz[:, 1], label='FL y', color='blue')
axes[0, 0].plot(t, fl_xyz[:, 2], label='FL z', color='magenta')
axes[0, 0].set_title('FL position ' + cpg.GAIT)
axes[0, 0].legend()

axes[1, 1].plot(t, rr_xyz[:, 0], label='RR x', color='red')
axes[1, 1].plot(t, rr_xyz[:, 1], label='RR y', color='blue')
axes[1, 1].plot(t, rr_xyz[:, 2], label='RR z', color='magenta')
axes[1, 1].set_title('RR position ' + cpg.GAIT)
axes[1, 1].legend()

axes[1, 0].plot(t, rl_xyz[:, 0], label='RL x', color='red')
axes[1, 0].plot(t, rl_xyz[:, 1], label='RL y', color='blue')
axes[1, 0].plot(t, rl_xyz[:, 2], label='RL z', color='magenta')
axes[1, 0].set_title('RL position ' + cpg.GAIT)
axes[1, 0].legend()

plt.tight_layout()
plt.show()

del fig
del axes

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,5))

axes[0, 1].plot(t, fr_joint[:, 0], label='FR q0', color='red')
axes[0, 1].plot(t, fr_xyz[:, 1], label='FR q1', color='blue')
axes[0, 1].plot(t, fr_xyz[:, 2], label='FR q2', color='magenta')
axes[0, 1].legend()
axes[0, 1].set_title('FR angles ' + cpg.GAIT)

axes[0, 0].plot(t, fl_joint[:, 0], label='FL q0', color='red')
axes[0, 0].plot(t, fl_xyz[:, 1], label='FL q1', color='blue')
axes[0, 0].plot(t, fl_xyz[:, 2], label='FL q2', color='magenta')
axes[0, 0].legend()
axes[0, 0].set_title('FL angles ' + cpg.GAIT)

axes[1, 1].plot(t, rr_joint[:, 0], label='RR q0', color='red')
axes[1, 1].plot(t, rr_joint[:, 1], label='RR q1', color='blue')
axes[1, 1].plot(t, rr_joint[:, 2], label='RR q2', color='magenta')
axes[1, 1].legend()
axes[1, 1].set_title('RR angles '+ cpg.GAIT)

axes[1, 0].plot(t, rl_joint[:, 0], label='RL q0', color='red')
axes[1, 0].plot(t, rl_joint[:, 1], label='RL q1', color='blue')
axes[1, 0].plot(t, rl_joint[:, 2], label='RL q2', color='magenta')
axes[1, 0].legend()
axes[1, 0].set_title('RL angles ' + cpg.GAIT)

plt.tight_layout()

plt.show()
