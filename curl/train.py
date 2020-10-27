import numpy as np
import torch
import dmc2gym
import json
import time
import copy
from utilities import *
from common import *
from video_object import *
from torchvision import transforms
from Actor_critic import *


def main():
    env = dmc2gym.make(domain_name=domain_name,task_name =task_name,seed = seed,
    visualize_reward=False,from_pixels=1,height=initial_image_size,width=initial_image_size)
    env.seed(seed1)
    env = Framestack(env,num_frames)
    video  = Video_object(video_dir,height,width,fps)
    device1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("hello")
    action_shape = env.action_space.shape
    #print(type(action_shape[-1]))
    obs_shape = (3*num_frame_stack,image_size,image_size)
    initial_obs_shape = (3*num_frame_stack,initial_image_size,initial_image_size)
    replay_buffer = Replay_buffer(capacity,batch_size)
    agent = final_agent(obs_shape,action_shape,hidden_dim,encoder_feature_dim,curl_latent_dim)

    episode,episode_reward,done = 0,0,True
    
    for step in range(num_epochs):
        print(step)
        #if steps%eval_steps==0:


        if done==True:

            obs = env.reset()
            next_obs = env.reset()
            done =False
            reward=0
            episode_step=0
            episode+=1
            action = env.action_space.sample()
            replay_buffer.append(make_tuple(obs,action,reward,next_obs,done))

        if step<initial_required_steps:
          action = env.action_space.sample()
          print(type(action))
          #print("initial_actions.format:{}",action)
        else:
          action = agent.select_action(obs)
          #action1 = action[0]
          print(action.shape)
          print(action)
          print(type(action))
          #print("actions_now.format:{}",action)
           
        #print(type(action))
        #print(action)
        #action = np.array(action)
        if step<initial_required_steps:
            #update_weights()
          loss1 = actor_loss(agent,obs)
          params1 = agent.actor.parameters()
          opt1 = torch.optim.Adam(params=params1,lr=lr)
          loss1.backward()
          opt1.step()
          loss2 = critic_loss(agent,obs,reward,next_obs,action,done)
          params11 = agent.critic.parameters()
          params22  =  agent.critic_target.parameters()
          opt2 = torch.optim.Adam(params = params11,lr=lr)
          loss2.backward()
          opt2.step()
          _,_,_,_,_,key,anchor = replay_buffer.sample_cpc(batch_size)
          loss3 = curl_loss(agent,key,anchor)
          params3 = agent.curl_agent.parameters()
          opt3 = torch.optim.Adam(params=params3,lr=lr)
          loss3.backward()
          opt3.step()
           


        next_obs,reward,done,_ = env.step(action)
        #done = 0 if episode_step+1==env.max_episode_steps    

        episode_reward+=reward
        replay_buffer.append(make_tuple(obs,action,reward,next_obs,done))

        obs = next_obs
        step+=1


if __name__=="__main__":
    main()