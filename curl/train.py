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
from torch.utils.tensorboard import SummaryWriter


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
    if not os.path.isdir(tensorboard_dir):
      os.mkdir(tensorboard_dir)
    writer  = SummaryWriter(tensorboard_dir)
    
    
    episode,episode_reward,done = 0,0,True
    prev_episode = 0
    curr_loss = 0
    best_reward = 0
    num_steps = 0
    for step in range(num_epochs):
        #print(step)
        #if steps%eval_steps==0:
        num_steps +=1

        if done==True:
            average_reward = episode_reward/num_steps
            print("______________________________________________________________________________________________________________________")
            print("Current Episode:{}".format(episode),"||","average_reward:{}".format(episode_reward/num_steps),"||","best_reward:{}".format(best_reward),"||","Current loss:{}".format(curr_loss),"||","number of steps in episode:{}".format(num_steps))
            obs = env.reset()
            next_obs = env.reset()
            done =False
            num_steps = 0
            episode_reward=0
            reward = 0
            best_reward = 0
            episode_step=0
            prev_episode = episode
            episode+=1
            action = env.action_space.sample()
            replay_buffer.append(make_tuple(obs,action,reward,next_obs,done))

        if step<initial_required_steps:
          action = env.action_space.sample()
          #print(type(action))
          #print("initial_actions.format:{}",action)
        else:
          action = agent.select_action(obs)
          #action1 = action[0]
          #print(action.shape)
          #print(action)
          #print(type(action))
          #print("actions_now.format:{}",action)
           
        #print(type(action))
        #print(action)
        #action = np.array(action)
        if step>initial_required_steps:
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
           

          curr_loss = loss1 + loss2 + loss3
        next_obs,reward,done,_ = env.step(action)
        if reward>best_reward:
          best_reward = reward
        #done = 0 if episode_step+1==env.max_episode_steps    

        episode_reward+=reward
        replay_buffer.append(make_tuple(obs,action,reward,next_obs,done))

        obs = next_obs
        step+=1
        writer.add_scalar("Average  Reward",episode_reward/(num_steps+1),episode)
        writer.add_scalar("Best  Reward",best_reward,episode)
        writer.add_scalar("Cumulative Loss",curr_loss,episode)
        
        if episode%10==0:
         if prev_episode!=episode:
          print("Saving Model............")
          current_ep = format(episode,'04')
          torch.save(agent.state_dict(),'model_' + current_ep + '.pth' )
          print("model has been saved")
          prev_episode = episode


if __name__=="__main__":
    main()
