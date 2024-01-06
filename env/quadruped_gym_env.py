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

import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = "SAC"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '121321105810'
log_dir = interm_dir + '123123101258'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training

env_config = {"motor_control_mode":"PD",
               "task_env": "FWD_LOCOMOTION", #  "LR_COURSE_TASK",
                "observation_space_mode": "OBS"}

# env_config = {}
env_config['render'] = False
env_config['record_video'] = False
env_config['add_noise'] = False 
env_config['competition_env'] = False

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
# plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# [TODO] initialize arrays to save data from simulation 
#
time_steps = 1000
speed = np.array(np.empty((0, 3)))
base_position = np.array(np.empty((0, 3)))
feet_arrays = np.empty((0, 4, 3))
contact = np.empty((0,4))
dist_goal = np.empty((0,1))

for i in range(time_steps):
    action, _states = model.predict(obs, deterministic=False)  # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0

    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
    #
    speed = np.append(speed, [env.envs[0].env.get_speed()], axis=0)
    base_position = np.append(base_position, [env.envs[0].env.robot.GetBasePosition()], axis=0)
    feet_array = np.array([env.envs[0].env.get_position_leg(j) for j in range(4)])
    feet_arrays = np.append(feet_arrays, [feet_array], axis=0)
    contact = np.append(contact, [env.envs[0].env.robot.GetContactInfo()[3]], axis=0)
    if env_config["task_env"]=="FLAGRUN":
        dist_goal = np.append(dist_goal, [env.envs[0].env.get_distance_and_angle_to_goal()[0]], axis=0)
    

time_steps = range(time_steps)
# [TODO] make plots:
plt.figure(figsize=(8, 4))
plt.plot(time_steps, speed[:, 0], label='speed x', color='blue')
plt.plot(time_steps, speed[:, 1], label='speed y', color='purple')
plt.plot(time_steps, speed[:, 2], label='speed z', color='orange')
plt.title('speeds')
plt.legend()
# plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time_steps, base_position[:, 0], label='x', color='blue')
plt.plot(time_steps, base_position[:, 1], label='y', color='purple')
plt.plot(time_steps, base_position[:, 2], label='z', color='orange')
if env_config["task_env"]=="FLAGRUN":
    plt.plot(time_steps, dist_goal[:], label='distance to goal', color='black')
plt.title('base position')
plt.legend()


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
axes[0, 1].plot(time_steps, contact[:, 0],color='black')
axes[0, 1].set_title('FR contact')

axes[0, 0].plot(time_steps, contact[:, 1], color='black')
axes[0, 0].set_title('FL contact')

axes[1, 1].plot(time_steps, contact[:, 2], color='black')
axes[1, 1].set_title('FL contact')

axes[1, 0].plot(time_steps, contact[:, 3], color='black')
axes[1, 0].set_title('RL contact')

plt.tight_layout()


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))

axes[0, 1].plot(time_steps, feet_arrays[:, 0, 1], label='FR y', color='blue')
axes[0, 1].plot(time_steps, feet_arrays[:, 0, 0], label='FR x', color='red')
axes[0, 1].plot(time_steps, feet_arrays[:, 0, 2], label='FR z', color='magenta')
axes[0, 1].set_title('FR position')
axes[0, 1].legend()

axes[0, 0].plot(time_steps, feet_arrays[:, 1, 0], label='FL x', color='red')
axes[0, 0].plot(time_steps, feet_arrays[:, 1, 1], label='FL y', color='blue')
axes[0, 0].plot(time_steps, feet_arrays[:, 1, 2], label='FL z', color='magenta')
axes[0, 0].set_title('FL position')
axes[0, 0].legend()

axes[1, 1].plot(time_steps, feet_arrays[:, 2, 0], label='RR x', color='red')
axes[1, 1].plot(time_steps, feet_arrays[:, 2, 1], label='RR y', color='blue')
axes[1, 1].plot(time_steps, feet_arrays[:, 2, 2], label='RR z', color='magenta')
axes[1, 1].set_title('RR position')
axes[1, 1].legend()

axes[1, 0].plot(time_steps, feet_arrays[:, 3, 0], label='RL x', color='red')
axes[1, 0].plot(time_steps, feet_arrays[:, 3, 1], label='RL y', color='blue')
axes[1, 0].plot(time_steps, feet_arrays[:, 3, 2], label='RL z', color='magenta')
axes[1, 0].set_title('RL position')
axes[1, 0].legend()

plt.tight_layout()
plt.show()
