import time
import os
import torch
import trimesh
import numpy as np
from tqdm import trange

from .generic_runner import GenericRunner

class CheckpointRunner(GenericRunner):

    def train_starts(self, model, experiment, checkpoint_dir):
        ## run a checkpoint iteration to save initialization and constant data
        ckpt_info = self.CKPTWrapper()
        #print(ckpt_info) 
        ckpt_info.checkpoint_dir     = os.path.join(checkpoint_dir, 'init/')
        self.run(model, experiment, ckpt_info)

    def run(self, model, experiment, ckpt_info):
        
        dataset = experiment['datasets']['train'].dataset

        for i in trange(dataset.num_checkpointing_samples()):

            #self.reset_surface_storage()
            
            #sample = dataset.get_checkpointing_sample(i)
            #name   = sample['name'][:-3].upper() #e.g. change 'spike_nA' into 'SPIKE'
    
            #self.checkpoint_sample(sample, model, experiment, ckpt_info)
            self.save_model(ckpt_info.checkpoint_dir, model, str(ckpt_info.epoch))
          
            models = [modelname for modelname in os.listdir(ckpt_info.checkpoint_dir+'/models') if modelname[:5]=='model']
            if len(models)>5:
                numbers = sorted([int(modelname[5:-4]) for modelname in models])
                os.remove(ckpt_info.checkpoint_dir+'/models/model'+str(numbers[0])+'.pth')
            
            self.end_checkpointing(model, ckpt_info)
            
    def save_model(self, checkpoint_folder, model, name=''):

        folder = checkpoint_folder+ '/models'
        #folder = '../checkpoints'
        file_name = 'model{}'.format(name)

        # save last model
        model_path = '{}/{}.pth'.format(folder, file_name)
        torch.save(model.state_dict(), model_path)
        

    #def checkpoint_sample(self, sample, model, experiment, ckpt_info):
    #    raise NotImplementedError()

    def end_checkpointing(self, model, ckpt_info):
        pass
        #self.save_model(ckpt_info.checkpoint_dir, model, 'finished_model')
