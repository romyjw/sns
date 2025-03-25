
import logging
import torch

from runners import CheckpointRunner
from tqdm import trange


class MCFCheckpointer(CheckpointRunner):

    def run(self, model, experiment, ckpt_info):
        pass
        '''dataset = experiment['datasets']['train'].dataset

        for i in trange(dataset.num_checkpointing_samples()):

            #self.reset_surface_storage()
            
            sample = dataset.get_checkpointing_sample(i)
            name   = sample['name'][:-3].upper() #e.g. change 'spike_nA' into 'SPIKE'
            print('name is', name)
            self.checkpoint_sample(sample, model, experiment, ckpt_info)

            self.save_model(ckpt_info.checkpoint_dir, model, str(ckpt_info.epoch))
            if os.path.exists( '../data/MCF/'+name):
                torch.save(model.state_dict(), '../data/MCF/'+name+'/current_weights.pth')

            models = [modelname for modelname in os.listdir(ckpt_info.checkpoint_dir+'/models') if modelname[:5]=='model']
            if len(models)>5:
                numbers = sorted([int(modelname[5:-4]) for modelname in models])
                os.remove(ckpt_info.checkpoint_dir+'/models/model'+str(numbers[0])+'.pth')
            
            self.end_checkpointing(model, name, ckpt_info)'''



