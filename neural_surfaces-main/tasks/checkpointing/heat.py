
import logging
import torch
import os

from runners import CheckpointRunner

class HeatCheckpointer(CheckpointRunner):

     
     def run(self, model, experiment, ckpt_info):
                
        #keep last 5 models
        self.save_model(ckpt_info.checkpoint_dir, model, str(ckpt_info.epoch))
        torch.save(model.state_dict(), '../data/heat/deepsdfCAMERA/current_field_weights.pth') #HARDCODED WARNING
        models = [modelname for modelname in os.listdir(ckpt_info.checkpoint_dir+'/models') if modelname[:5]=='model']
        if len(models)>5:
            numbers = sorted([int(modelname[5:-4]) for modelname in models])
            os.remove(ckpt_info.checkpoint_dir+'/models/model'+str(numbers[0])+'.pth')


