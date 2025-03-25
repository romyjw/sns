import torch
from math import ceil

from sample_preparation.surface_sampling import sample_surface

from .mixin import DatasetMixin
import numpy as np

class ModelDataset(DatasetMixin):

    def __init__(self, config):
        print('modeldataset init')
        self.sample_path = config['sample_path']
        self.num_points  = config['num_points']

        self.sample = self.read_sample(self.sample_path)
        

        ## read sample split into patches (look only at global)
                    
        self.points        = self.sample['points'].float()
        self.param         = self.sample['param'].float()
        self.faces         = self.sample['faces'].long()
        if not self.sample['normals'] is None:
            self.normals       = self.sample['normals'].float()
        else:
            self.normals = None
        self.name          = self.sample['name']
            
        ## split into batches
        self.num_batches = ceil(self.points.size(0) / self.num_points)
        self.batchs_idx = self.split_to_blocks(self.points.size(0), self.num_batches)

        ## check how mask normals
        self.mask_normals = True if 'mask_normals' not in config else config['mask_normals']
        
    def __len__(self):
        return self.num_batches


    def __getitem__(self, index):
        
        print('getting a batch')

        ## get current batch
        idx     = self.batchs_idx[index%self.num_batches]
        points  = self.points[idx]
        params  = self.param[idx]
        normals = self.normals[idx]
        N = normals.size(0)
        # print(N, self.num_points)        
        print(points.shape, params.shape, normals.shape)        

        ## leave it as just normals for now


        ## sample extra points from the surface
        params_to_sample = [self.param]
        new_points, new_normals, new_params = sample_surface(self.num_points, self.points,
                                                self.faces, params_to_sample, method='pytorch3d')

        ## concat sampled points
        params  = torch.cat([params, new_params[0]], dim=0)
        points  = torch.cat([points, new_points], dim=0)
        normals = torch.cat([normals, new_normals], dim=0)
        print(points.shape, params.shape, normals.shape)     

        ## mask normals
        mask = torch.ones(params.size(0)).bool()
        if self.mask_normals:
            mask[:] = True ###### True by default
            mask[:N] = False ###### False for the points that are exactly on vertices (original points)
        
                
        data_dict = {
                'param':   params,
                'gt':      points,
                'normals': normals,
                'mask':    mask
        }

        return data_dict


    def num_checkpointing_samples(self):
        return 1


    def get_checkpointing_sample(self, index):

        data_dict = {}
        data_dict['param'] = self.param
        data_dict['gts']   = self.points
        data_dict['faces'] = self.faces
        data_dict['name']  = self.name

        return data_dict


class ModelDatasetNoFaces(DatasetMixin):

    def __init__(self, config):
        print('modeldataset init (no faces version)')
        self.sample_path = config['sample_path']
        self.num_points  = config['num_points']

        self.sample = self.read_sample(self.sample_path)
        
        self.points        = self.sample['points'].float()
        self.param         = self.sample['param'].float()
        if not self.sample['normals'] is None:
            self.normals       = self.sample['normals'].float()
        else:
            self.normals = None
        self.name          = self.sample['name']
            
           

        ## split into batches
        self.num_batches = ceil(self.points.size(0) / self.num_points)
        print('points size, num_points per batch',self.points.size(0), self.num_points)
        self.batchs_idx = self.split_to_blocks(self.points.size(0), self.num_batches)

        ## check how mask normals
        self.mask_normals = True if 'mask_normals' not in config else config['mask_normals']



    def __len__(self):
        return self.num_batches


    def __getitem__(self, index):

        ## get current batch
        idx     = self.batchs_idx[index%self.num_batches]
        points  = self.points[idx]
        params  = self.param[idx]
        normals = self.normals[idx]
        N = normals.size(0)

        ## mask normals
        mask = torch.ones(params.size(0)).bool()
        if self.mask_normals:
            
                mask[:] = True
                mask[:N] = False ######    ??????? give it a go.
        
    
        print(points.shape, params.shape, normals.shape)        
        data_dict = {
                'param':   params,
                'gt':      points,
                'normals': normals,
                'mask':    mask
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
