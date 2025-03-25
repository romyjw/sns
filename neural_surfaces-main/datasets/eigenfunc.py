
import torch
from math import ceil

from sample_preparation.surface_sampling import sample_surface

from .mixin import DatasetMixin
import numpy as np

class EigenfuncDataset(DatasetMixin):

    def __init__(self, config):
        print('eignfunc init')
        self.sample_path = config['sample_path']
        self.num_points  = config['num_points']

        self.sample = self.read_sample(self.sample_path)
    
        ## read sample split into patches (look only at global)
                    
        self.points        = self.sample['points'].float()
        self.param         = self.sample['param'].float()
        self.normals       = self.sample['normals'].float()
        try:
            self.transpose_inverse_jacobians = self.sample['transpose_inverse_jacobians'].float()
        except:
            self.transpose_inverse_jacobians = None
        try:
            self.ortho_functions = self.sample['ortho_functions'] #self.sample['transpose_inverse_jacobians'].float()
        except:
            print('no ortho functions present in sample')
            self.ortho_functions = []
        
        
        #print(self.points.shape,'points shape')
        self.name          = self.sample['name']
            
        ## split into batches
        self.num_batches = ceil(self.points.size(0) / self.num_points)
        self.batchs_idx = self.split_to_blocks(self.points.size(0), self.num_batches)

    
    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):

        # get current batch
        idx     = self.batchs_idx[index%self.num_batches]
        points  = self.points[idx]
        params  = self.param[idx]
        normals = self.normals[idx]
        transpose_inverse_jacobians = self.transpose_inverse_jacobians[idx,:,:]
        
        ortho_functions = [ ortho_function[idx].float() for ortho_function in self.ortho_functions ] 
        
           
        data_dict = {
                'param':   params,
                'gt':      points,
                'normals': normals,
                'transpose_inverse_jacobians' : transpose_inverse_jacobians,
                'ortho_functions' : ortho_functions
                        }

        return data_dict


    def num_checkpointing_samples(self):
        return 1


    def get_checkpointing_sample(self, index):

        data_dict = {}
        data_dict['param'] = self.param
        data_dict['gts']   = self.points
        data_dict['name']  = self.name

        return data_dict


class EigenfuncOverfitDataset(DatasetMixin):

    def __init__(self, config):
        print('eignfunc init')
        self.sample_path = config['sample_path']
        self.num_points  = config['num_points']

        self.sample = self.read_sample(self.sample_path)
        

        ## read sample split into patches (look only at global)
                    
        self.points        = self.sample['points'].float()
        self.param         = self.sample['param'].float()
        self.discrete_eigenfunc = self.sample['discrete_eigenfunc'].float()
        
        
        #print(self.points.shape,'points shape')
        self.name          = self.sample['name']
            
        ## split into batches
        self.num_batches = ceil(self.points.size(0) / self.num_points)
        self.batchs_idx = self.split_to_blocks(self.points.size(0), self.num_batches)

        

    def __len__(self):
        return self.num_batches


    def __getitem__(self, index):

        # get current batch
        idx     = self.batchs_idx[index%self.num_batches]
        points  = self.points[idx]
        params  = self.param[idx]
        discrete_eigenfunc = self.discrete_eigenfunc[idx]
        
        

                
        data_dict = {
                'param':   params,
                'gt':      points,
                'discrete_eigenfunc': discrete_eigenfunc
                        }

        return data_dict


    def num_checkpointing_samples(self):
        return 1


    def get_checkpointing_sample(self, index):
        ##I think, this isn't used.

        data_dict = {}
        data_dict['param'] = self.param
        data_dict['gts']   = self.points
        data_dict['name']  = self.name

      

        return data_dict
