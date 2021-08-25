#!/usr/bin/env python
#-*- coding:UTF-8 -*-

import rospy
import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray, Float32
from environment_actual import Env
from environment_actual import ACTION_V_MAX, ACTION_W_MAX
import torch
import torch.nn.functional as F
import gc
import torch.nn as nn
import math
from collections import deque
import copy
import time

#---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))
dirPath = dirPath.replace('DDPG_NAV/src/', 'DDPG_NAV/models/')
#---Functions to make network updates---#
 
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data*(1.0 - tau)+ param.data*tau)

def hard_update(target,source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

#---Ornstein-Uhlenbeck Noise for action---#
 
class OUNoise(object):
    def __init__(self, action_space, mu=[0.01, 0], theta=[0.02, 0.08], dt=1, max_sigma=[0.99, 0.99], min_sigma=[0.1, 0.1], decay_period= 800000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space
        self.dt = dt
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.randn(self.action_dim) * math .sqrt(self.dt)
        self.state = x + dx
        return self.state
    
    def get_noise(self, t=0): 
        ou_state = self.evolve_state()
        # print('noise' + str(ou_state))
        decaying = float(float(t)/ self.decay_period)
        self.sigma[0] = max(self.sigma[0] - (self.max_sigma[0] - self.min_sigma[0]) * min(1.0, decaying), self.min_sigma[0])
        self.sigma[1] = max(self.sigma[1] - (self.max_sigma[1] - self.min_sigma[1]) * min(1.0, decaying), self.min_sigma[1])
        return ou_state

#---Critic--#

EPS = 0.003
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1./np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v,v)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(state_dim, 125)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        # self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        
        self.fa1 = nn.Linear(action_dim, 125)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)
        # self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())
        
        self.fca1 = nn.Linear(250, 250)
        nn.init.xavier_uniform_(self.fca1.weight)
        self.fca1.bias.data.fill_(0.01)
        # self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        
        self.fca2 = nn.Linear(250, 1)
        nn.init.xavier_uniform_(self.fca2.weight)
        self.fca2.bias.data.fill_(0.01)
        # self.fca2.weight.data.uniform_(-EPS, EPS)
        
    def forward(self, state, action):
        xs = torch.relu(self.fc1(state))
        xa = torch.relu(self.fa1(action))
        x = torch.cat((xs,xa), dim=1)
        x = torch.relu(self.fca1(x))
        vs = self.fca2(x)
        return vs

#---Actor---#

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w):
        super(Actor, self).__init__()
        self.state_dim = state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w
        
        self.fa1 = nn.Linear(state_dim, 250)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)
        # self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())
        
        self.fa2 = nn.Linear(250, 250)
        nn.init.xavier_uniform_(self.fa2.weight)
        self.fa2.bias.data.fill_(0.01)
        # self.fa2.weight.data = fanin_init(self.fa2.weight.data.size())
        
        self.fa3 = nn.Linear(250, action_dim)
        nn.init.xavier_uniform_(self.fa3.weight)
        self.fa3.bias.data.fill_(0.01)
        # self.fa3.weight.data.uniform_(-EPS,EPS)
        
    def forward(self, state):
        x = torch.relu(self.fa1(state))
        x = torch.relu(self.fa2(x))
        action = self.fa3(x)
        if state.shape <= torch.Size([self.state_dim]):
            action[0] = torch.tanh(action[0])*self.action_limit_v
            action[1] = torch.tanh(action[1])*self.action_limit_w
        else:
            action[:,0] = torch.tanh(action[:,0])*self.action_limit_v
            action[:,1] = torch.tanh(action[:,1])*self.action_limit_w
        return action

#---Memory Buffer---#

class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0
        
    def sample(self, count):
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)
        
        s_array = np.float32([array[0] for array in batch])
        a_array = np.float32([array[1] for array in batch])
        r_array = np.float32([array[2] for array in batch])
        new_s_array = np.float32([array[3] for array in batch])
        done_array = np.float32([array[4] for array in batch])
        
        return s_array, a_array, r_array, new_s_array, done_array
    
    def len(self):
        return self.len
    
    def add(self, s, a, r, new_s, done):
        transition = (s, a, r, new_s, done)
        self.len += 1 
        if self.len > self.maxSize:
            self.len = self.maxSize
            del self.buffer[0]
        self.buffer.append(transition)

#---Where the train is made---#

BATCH_SIZE = 512
LEARNING_RATE = 0.001
TAU = 0.001

class Trainer:
    
    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w, ram):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w
        #print('w',self.action_limit_w)
        self.ram = ram
        #self.iter = 0 
        
        self.actor = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE)
        
        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LEARNING_RATE)
        self.pub_qvalue = rospy.Publisher('qvalue', Float32, queue_size=5)
        self.qvalue = 0.0
        self.loss_actor_value = 0.0
        self.loss_critic_value = 0.0
        self.count = 0
        self.t = 0
        self.GAMMA_INITIAL = 0.99
        self.GAMMA_END = 0.8
        self.GAMMA = 0.99
        self.GAMMA_DECAY_PERIOD = 600000

        # self.pub_trainning_msgs = rospy.Publisher('trainning_msgs', Float32MultiArray, queue_size=5)
        # self.trainning_msgs = Float32MultiArray()

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        
    def get_exploitation_action(self,state):
        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()
        #print('actionploi', action)
        return action.data.numpy()
        
    def get_exploration_action(self, state):
        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()
        #noise = self.noise.sample()
        #print('noisea', noise)
        #noise[0] = noise[0]*self.action_limit_v
        #noise[1] = noise[1]*self.action_limit_w
        #print('noise', noise)
        new_action = action.data.numpy() #+ noise
        #print('action_no', new_action)
        return new_action
    
    def optimizer(self):
        s_sample, a_sample, r_sample, new_s_sample, done_sample = ram.sample(BATCH_SIZE)
        
        s_sample = torch.from_numpy(s_sample)
        a_sample = torch.from_numpy(a_sample)
        r_sample = torch.from_numpy(r_sample)
        new_s_sample = torch.from_numpy(new_s_sample)
        done_sample = torch.from_numpy(done_sample)
        #-------------- optimize critic
        
        a_target = self.target_actor.forward(new_s_sample).detach()
        next_value = torch.squeeze(self.target_critic.forward(new_s_sample, a_target).detach())
        # y_exp = r + gamma*Q'(s', P'(s')) 
        y_expected = r_sample + (1 - done_sample) * self.GAMMA * next_value
        self.GAMMA = self.GAMMA_INITIAL-((self.GAMMA_INITIAL-self.GAMMA_END)*float(float(self.t)/ self.GAMMA_DECAY_PERIOD))
        self.t +=1
        # y_pred = Q(s,a)
        y_predicted = torch.squeeze(self.critic.forward(s_sample, a_sample))
        #-------Publisher of Vs------
        self.qvalue = y_predicted.detach()
        # self.pub_qvalue.publish(torch.max(self.qvalue))
        # print(self.qvalue, torch.max(self.qvalue))
        #----------------------------
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.loss_critic_value = loss_critic.detach()
        
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        
        #------------ optimize actor
        pred_a_sample = self.actor.forward(s_sample)
        loss_actor = -1*torch.mean(self.critic.forward(s_sample, pred_a_sample))
        self.loss_actor_value = loss_actor.detach()

        # self.trainning_msgs.data = [torch.max(self.qvalue), loss_actor.detach(), loss_critic.detach()]
        # self.pub_trainning_msgs.publish(self.trainning_msgs)

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # self.count += 1
        # if self.count==20:
        #     self.count = 0
        soft_update(self.target_actor, self.actor, TAU)
        soft_update(self.target_critic, self.critic, TAU)
            # hard_update(self.target_actor, self.actor)
            # hard_update(self.target_critic, self.critic)
    
    def save_models(self, episode_count):
        torch.save(self.target_actor.state_dict(), dirPath + str(episode) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), dirPath + str(episode_count) + '_critic.pt')
        print('****Models saved***')
        
    def load_models(self, episode):
        self.actor.load_state_dict(torch.load(dirPath + str(episode)+ '_actor.pt'))
        self.critic.load_state_dict(torch.load(dirPath + str(episode)+ '_critic.pt'))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print('***Models load***')



#---Run agent---#

is_training = False
is_load_model = True
load_ep = 0

exploration_decay_rate = 0.001

MAX_EPISODES = 10001
MAX_STEPS = 500+1
MAX_BUFFER = 100000
rewards_all_episodes = []

STATE_DIMENSION = 28
ACTION_DIMENSION = 2
# ACTION_V_MAX = 0.3 # m/s
# ACTION_W_MAX = 2. # rad/s
world = 'stage_4'
if is_training:
    var_v = ACTION_V_MAX*.5
    var_w = ACTION_W_MAX*2*.5
else:
    var_v = ACTION_V_MAX*0.10
    var_w = ACTION_W_MAX*0.10

print('State Dimensions: ' + str(STATE_DIMENSION))
print('Action Dimensions: ' + str(ACTION_DIMENSION))
print('Action Max: ' + str(ACTION_V_MAX) + ' m/s and ' + str(ACTION_W_MAX) + ' rad/s')
ram = MemoryBuffer(MAX_BUFFER)
trainer = Trainer(STATE_DIMENSION, ACTION_DIMENSION, ACTION_V_MAX, ACTION_W_MAX, ram)
noise = OUNoise(ACTION_DIMENSION, max_sigma=[0.6, 0.6] , min_sigma=[0.1, 0.1], decay_period=8000000)

if is_load_model:
    trainer.load_models(load_ep)

if __name__ == '__main__':
    rospy.init_node('actual_test')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    env = Env(action_dim=ACTION_DIMENSION)
    before_training = 4

    past_action = np.zeros(ACTION_DIMENSION)

    for ep in range(MAX_EPISODES):
        if is_load_model:
            ep += (load_ep+1)
        done = False
        state = env.reset()
        if is_training and not ep%10 == 0 and ram.len >= before_training*MAX_STEPS:
            print('---------------------------------')
            print('Episode: ' + str(ep) + ' training')
            print('---------------------------------')
        else:
            if ram.len >= before_training*MAX_STEPS:
                print('---------------------------------')
                print('Episode: ' + str(ep) + ' evaluating')
                print('---------------------------------')
            else:
                print('---------------------------------')
                print('Episode: ' + str(ep) + ' adding to memory')
                print('---------------------------------')

        rewards_current_episode = 0.

        time_end, time_start = 0, 0
        for step in range(MAX_STEPS):
            
            time_end = time.time()
            time_c= time_end - time_start
            # print('time cost', time_c, 's')
            time_start = time.time()

            state = np.float32(state)

            # if is_training and not ep%10 == 0 and ram.len >= before_training*MAX_STEPS:

            if is_training and (not ep%10 == 0 or ep==0):
                action = trainer.get_exploration_action(state)
                N = copy.deepcopy(noise.get_noise(t=step))
                N[0] = N[0]*ACTION_V_MAX
                N[1] = N[1]*ACTION_W_MAX
                action[0] = np.clip(action[0] + N[0], -ACTION_V_MAX, ACTION_V_MAX)
                action[1] = np.clip(action[1] + N[1], -ACTION_W_MAX, ACTION_W_MAX)

            else:
                action = trainer.get_exploration_action(state)


            if not is_training:
                action = trainer.get_exploitation_action(state)
            next_state, reward, done = env.step(action, past_action)
            # print('action', action,'r',reward)
            past_action = copy.deepcopy(action)
            
            rewards_current_episode += reward
            next_state = np.float32(next_state)

            if not ep%10 == 0 or not ram.len >= before_training*MAX_STEPS:

                if reward >= 100.:
                    print('***\n-------- Maximum Reward ----------\n****')
                    for _ in range(10):
                        ram.add(state, action, reward, next_state, done)

                else:
                    ram.add(state, action, reward, next_state, done)
            state = copy.deepcopy(next_state)
            
            flag = False
            if ram.len >= before_training*MAX_STEPS and is_training and not ep%10 == 0:
                trainer.optimizer()
                flag = True

            if (done or step == MAX_STEPS-1) :
                print('reward per ep: ' + str(rewards_current_episode))
                print('*\nbreak step: ' + str(step) + '\n*')
                # print('explore_v: ' + str(var_v) + ' and explore_w: ' + str(var_w))
                print('sigma: ' + str(noise.sigma))
                # rewards_all_episodes.append(rewards_current_episode)
                print("lenth of memorybuffer "+str( ram.len))

                if flag:
                    total_reward = rewards_current_episode
                    result.data = [total_reward, torch.median(trainer.qvalue).item(), trainer.loss_actor_value.item(), trainer.loss_critic_value.item()]
                    pub_result.publish(result)
                break
        if ep%20 == 0 and ep!=0 and is_training:
            trainer.save_models(ep)

print('Completed Training')
