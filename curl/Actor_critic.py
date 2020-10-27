import torch
import  torch.nn as nn
import torch.nn.functional as  F
import os
from common import * 
from utilities  import *
from encode_image import *

#gaussian_logprob function is copied from the official code"


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)
class Actor(nn.Module):
    def __init__(self,num_input_features,action_space,hidden_dim,log_std_min,log_std_max):
        super(Actor,self).__init__()
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_size = action_space
        self.fc1 = nn.Linear(num_input_features,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,hidden_dim)
        self.fc4 = nn.Linear(hidden_dim,2*action_space[-1])
        self.act_layer = F.relu

    def forward(self,x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.fc2(x)
        x = self.act_layer(x)
        x = self.fc3(x)
        x = self.act_layer(x)
        x = self.fc4(x)
        mu,log_std = torch.chunk(x,2,dim=-1)
        print("shape_check:{}".format(mu.shape))
        controlled_log_std = self.log_std_min + (self.log_std_max-self.log_std_min)*(log_std+1)
        pi = torch.exp(controlled_log_std)*torch.rand_like(mu) + mu 
        log_pi = gaussian_logprob(torch.randn_like(mu),controlled_log_std)
        return mu,controlled_log_std,pi,log_pi
        
class Critic(nn.Module):
    def __init__(self,num_input_feature,hidden_dim,action_dim):
        super(Critic,self).__init__()
        self.encoder = Encoder(3,32,32,32)
        self.hidden_dim = hidden_dim
        self.input_dim = num_input_feature
        self.action_size = action_dim[-1]
        self.fc1 = nn.Linear(num_input_feature+action_dim[-1],hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,hidden_dim)
        self.fc4 = nn.Linear(hidden_dim,1)
        self.act_layer = F.relu
        self.fc11 = nn.Linear(num_input_feature+action_dim[-1],hidden_dim)
        self.fc22 = nn.Linear(hidden_dim,hidden_dim)
        self.fc33 = nn.Linear(hidden_dim,hidden_dim)
        self.fc44 = nn.Linear(hidden_dim,1)
    def forward(self,x,x1):
        x1 = x1.view(-1)
        x = x.view(-1)
        x = torch.cat([x,x1],dim=-1)
        y = self.fc1(x)
        y = self.act_layer(y)
        y = self.fc2(y)
        y = self.act_layer(y)
        y = self.fc3(y)
        y = self.act_layer(y)
        y = self.fc4(y)
        y = self.act_layer(y)
        y1 = self.fc11(x)
        y1 = self.act_layer(y1)
        y1 = self.fc22(y1)
        y1 = self.act_layer(y1)
        y1 = self.fc33(y1)
        y1 = self.act_layer(y1)
        y1 = self.fc44(y1)
        y1 = self.act_layer(y1)
        
        return y,y1


class Curl(nn.Module):
    def __init__(self,z_dim,batch_size,encoder_target,encoder):

        super(Curl,self).__init__()
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.encoder_target = encoder_target
        self.encoder = encoder
        self.weights = nn.Parameter(torch.zeros(z_dim,z_dim))
    def forward(self,x,target=False):
        if target:
            with torch.no_grad():
             z = self.encoder_target(x)
        else:
            z = self.encoder(x)

        return z
    

    def logit_conversion(self,z_anchors,z_targets):
        y = torch.matmul(self.weights,z_targets.T)
        logits =  torch.matmul(z_anchors,y)
        logits -= torch.max(logits,1)[0].unsqueeze(dim=1)
        return logits


class final_agent(nn.Module):
    def __init__(self,observation,action_size,hidden_dim,encoder_feature_size,curl_latent_dim):
        super(final_agent,self).__init__()
        self.observation_size = observation
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.encoder_feature_size = encoder_feature_size
        self.curl_latent_dim = curl_latent_dim
        self.critic = Critic(encoder_feature_size,hidden_dim,action_size).to(device)
        self.critic_target = Critic(encoder_feature_size,hidden_dim,action_size).to(device)
        self.actor = Actor(encoder_feature_size,action_size,hidden_dim,log_std_min,log_std_max).to(device)
        self.curl_agent = Curl(curl_latent_dim,batch_size,self.critic.encoder,self.critic_target.encoder).to(device)
        self.loss_function = nn.CrossEntropyLoss()
    
    def select_action(self,obs):
        obs = torch.FloatTensor(obs).to(device)
        obs = obs[None,:]
        obs = self.critic.encoder(obs)

        #if obs.shape[-1] != self.observation_size:
         #obs = center_crop  (obs,target_image_size)
        with torch.no_grad():
          #obs = torch.FloatTensor(obs).to(device)
          #obs = obs.unsqueeze(0) 
          mu,_,pi,_  = self.actor(obs)
          pi = torch.tanh(pi)
        #return mu.cpu().data.numpy().flatten(),pi.cpu().data.numpy()
        return pi.cpu().data.numpy()[0]

    
    

    
    


        







        


        


        





         
        
        
         
