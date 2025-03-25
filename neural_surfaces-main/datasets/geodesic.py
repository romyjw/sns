
import torch
from math import ceil

from sample_preparation.surface_sampling import sample_surface

from .mixin import DatasetMixin
import numpy as np

class GeodesicDataset(DatasetMixin):

    def __init__(self, config):
        print('eignfunc init')
        self.sample_path = config['sample_path']
        self.num_points  = config['num_points']

        self.sample = self.read_sample(self.sample_path)
    
        ## read sample split into patches (look only at global)
                    
        self.param         = self.sample['param'].float()


        
        
        #print(self.points.shape,'points shape')
        self.name          = self.sample['name']
            
        ## split into batches
        self.num_batches = ceil(self.param.size(0) / self.num_points)
        self.batchs_idx = self.split_to_blocks(self.param.size(0), self.num_batches)

    
    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):

        # get current batch
        idx     = self.batchs_idx[index%self.num_batches]
        
        params  = self.param[idx]
        
           
        data_dict = {
                'param':   params
                            }

        return data_dict


    def num_checkpointing_samples(self):
        return 1


    def get_checkpointing_sample(self, index):

        data_dict = {}
        data_dict['param'] = self.param
        data_dict['name']  = self.name

        return data_dict
