import torch
import numpy as np
import gym 
import os
from torch.utils.data import Dataset,DataLoader
import time
from gym.spaces import box
import collections
from gym import wrappers
from collections import deque
from gym.spaces import Box
from common import *   
import torch.nn.functional as F
from skimage.util.shape import view_as_windows

Experience = collections.namedtuple('Experience',field_names=['state','action','reward','done','new_state'])

class Replay_buffer(object):
    def __init__(self,capacity,batch_size):
        self.buffer  = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)
    
    def append(self,experience):
        self.buffer.append(experience)
    
    def sample(self,batch_size):
        indices = np.random.choice(len(self.buffer),batch_size,replace=False)
        
        states = [self.buffer[idx][0]for idx in indices]
        actions = [self.buffer[idx][0]for idx in indices]
        rewards = [self.buffer[idx][0]for idx in indices]
        dones = [self.buffer[idx][0]for idx in indices]
        next_states = [self.buffer[idx][0]for idx in indices]
        
        #states,actions,rewards,dones,next_states = [self.buffer[idx] for idx in indices]

        return np.array(states),np.array(actions),
        np.array(rewards,dtype =np.float32),
        np.array(dones,dtype =np.uint8),
        np.array(next_states)

    def sample_cpc(self,batch_size):

        indices = np.random.choice(len(self.buffer),batch_size,replace=True)
        
        states = [self.buffer[idx][0]for idx in indices]
        actions = [self.buffer[idx][1]for idx in indices]
        rewards = [self.buffer[idx][2]for idx in indices]
        dones = [self.buffer[idx][3]for idx in indices]
        next_states = [self.buffer[idx][4]for idx in indices]
        #states,actions,rewards,dones,next_states = [self.buffer[idx] for idx in indices]
        key = states
        #anchor = random_crop(states,image_size)
        anchor = key

        return np.array(states),np.array(actions),np.array(rewards,dtype =np.float32),np.array(dones,dtype =np.uint8),np.array(next_states),np.array(anchor),np.array(key)

        


class Framestack(gym.Wrapper):
    def __init__(self,env,num_stack):
        super(Framestack,self).__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        self.env = env
        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)
        self.maximum_episodes = env._max_episode_steps

    def get_abs(self):
        assert len(self.frames)==self.num_stack
        return np.concatenate(list(self.frames),axis=0)
    def reset(self):
        obs = self.env.reset()
        for i in range(self.num_stack):
            self.frames.append(obs)
        return self.get_abs()
    def step(self,action):
        obs,reward,done,info = self.env.step(action)
        self.frames.append(obs)
        return self.get_abs(),reward,done,info

def center_crop(image,output_size):
    h,w = image.shape[1:]
    top = (h-output_size)//2
    left = (h-output_size)//2
    image = image[:,top:top+output_size,left:left+output_size]
    return image
    
def random_crop(imgs, output_size):
    imgs= np.array(imgs)
    
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    #crop_max = img_size - output_size + 30
    crop_max = 15
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs

def make_tuple(obs,action,reward,next_obs,done):
    a  = []
    a.append(obs)
    a.append(action)
    a.append(reward)
    a.append(next_obs)
    a.append(done)
    
    return tuple(a)

def critic_loss(agent,obs,reward,next_obs,action,not_done):
        next_obs = torch.tensor(next_obs)
        next_obs = next_obs.to(device)
        next_obs = next_obs.float()
        obs = torch.tensor(obs)
        obs = obs.to(device)
        obs = obs.float()
        obs = obs[None,:]
        obs = agent.critic.encoder(obs)
        action = torch.tensor(action)
        action = action.to(device)
        next_obs = next_obs[None,:]
        next_obs = agent.critic.encoder(next_obs)
        with torch.no_grad():
            mu,policy,log_policy,_ = agent.actor(next_obs)
            target_1,target_2=  agent.critic(next_obs,policy)
            #target_v = torch.min(target_1,target_2) - self.alpha*log_policy
            target_v = torch.min(target_1,target_2) - lr*log_policy
            target_q = reward + (not_done*discount_factor*target_v)
        curr_q1,curr_q2 = agent.critic(obs,action)
        agent.critic_loss= F.mse_loss(curr_q1,target_q) + F.mse_loss(curr_q2,target_q)
        return agent.critic_loss
        
def actor_loss(agent,obs):
        
        obs = torch.tensor(obs)
        obs = obs.to(device)
        obs = obs.float()
        obs = obs[None,:]
        obs = agent.critic.encoder(obs)
        
        mu,policy,log_policy,_ = agent.actor(obs)
        actor_target_1,actor_target_2  = agent.critic(obs,policy)
        actor_Q = torch.min(actor_target_1,actor_target_2)
        #agent.actor_loss = (agent.alpha.detach()*log_policy-actor_Q).mean()
        agent.actor_loss = (log_policy-actor_Q).mean()
        return agent.actor_loss

def curl_loss(agent,anchor,keys):
        anchor = torch.tensor(anchor)
        keys = torch.tensor(keys)
        anchor = anchor.float()
        keys = keys.float()
        anchor = anchor.to(device)
        keys = keys.to(device)
        z_anchor = agent.curl_agent(anchor,target=False)
        z_key  = agent.curl_agent(keys,target=False)
        logits = agent.curl_agent.logit_conversion(z_anchor,z_key)
        labels = torch.arange(logits.shape[0]).long().to(device)
        agent.curl_loss = agent.loss_function(logits,labels)
        return agent.curl_loss

def random_crop_list(images,output_size):
    for image in images:
        image = random_crop(image,output_size)
    
    return images