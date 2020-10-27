import os
import numpy as np
from PIL import Image
class Video_object(object):
    def __init__(self,dir_name,height,width,fps):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.fps = fps
        self.frame_list = []
    
    def record(self,env):
        frame = env.render(mode='rgb_array',height=self.height,width=self.width,camera_id = 0)
        self.frame_list.append(frame)

    def save(self,file_name):
        file_path = os.path.join(dir_name,file_name)
        Image.save(file_path,self.frames,fps=self.fps)