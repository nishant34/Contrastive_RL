import torch
import numpy as np
import os


domain_name = 'cheetah'
task_name =  'run'
image_size = 84
initial_image_size = 32
init_steps = 5
num_train_steps = 80
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 1
seed1 = 1
num_frames = 20
num_frame_stack=20
video_dir = 'video1/'
height =64
width = 64
fps = 30
capacity = 50
batch_size = 10
hidden_dim = 32
encoder_feature_dim = 32
curl_latent_dim = 32
encoder_feature_size = 32
log_std_min = -0.01
log_std_max = 0.01
num_epochs = 20
initial_required_steps = 10
target_image_size = 32
lr = 0.00001
discount_factor = 0.15
