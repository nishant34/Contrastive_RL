import numpy as  np 
import  torch.nn as nn
import torch.nn.functional as F
import os
import torch
from common import *
from utilities import *


class Encoder(nn.Module):
    def __init__(self,Image_channels,middle_dim,Encoded_feature_size,num_filters):
            super(Encoder,self).__init__()
            self.num_input_channels = Image_channels*num_frame_stack
            self.output_size = Encoded_feature_size
            self.num_filters = num_filters
            self.conv1 = nn.Conv2d(self.num_input_channels,num_filters,3,stride=1)
            self.bn = nn.BatchNorm2d(num_filters)
            self.conv2 = nn.Conv2d(self.num_filters,num_filters,3,stride=1)
            self.conv3 = nn.Conv2d(self.num_filters,num_filters,3,stride=1)
            self.conv4 = nn.Conv2d(self.num_filters,num_filters,3,stride=2)
            self.conv5 = nn.Conv2d(self.num_filters,num_filters,3,stride=1)
            #self.final_fc = nn.Linear(num_filters*middle_dim*middle_dim,self.output_size)
            self.final_fc = nn.Linear(3200,self.output_size)

    def distribtuion(self,mu,logstd):
        return torch.exp(logstd)*torch.randn_like(logstd) + mu
    
    def forward(self,obs):
        obs = obs/255
        obs = self.conv1(obs)
        obs = self.bn(obs)
        obs  = F.relu(obs)
        obs = self.conv2(obs)
        obs = self.bn(obs)
        obs  = F.relu(obs)
        obs = self.conv3(obs)
        obs = self.bn(obs)
        obs  = F.relu(obs)
        obs = self.conv4(obs)
        obs = self.bn(obs)
        obs  = F.relu(obs)
        obs = self.conv5(obs)
        obs = self.bn(obs)
        out  =obs.view(obs.shape[0],-1)
        out = self.final_fc(out)
        out  = torch.tanh(out)
        return out
    
            

