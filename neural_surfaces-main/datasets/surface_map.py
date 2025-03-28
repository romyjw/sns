
import numpy as np
import torch

from sample_preparation.surface_sampling import sample_surface

import trimesh

from .mixin import DatasetMixin


class SurfaceMapDataset(DatasetMixin):

    def __init__(self, config):

        self.config     = config
        #print('config',config)
        self.num_points = config['num_points']

        self.sample_source = self.read_sample(config['sample_source'])
        self.sample_target = self.read_sample(config['sample_target'])

        ## extract all data
        self.source_uvs       = self.sample_source['param']
        self.source_faces     = self.sample_source['faces']
        self.source_points    = self.sample_source['points']
        self.source_C         = self.sample_source['C'] # normalization constant
        #self.source_boundary  = self.sample_source['boundary']
        self.source_name      = self.sample_source['name']

        self.target_uvs       = self.sample_target['param']
        self.target_faces     = self.sample_target['faces']
        self.target_points    = self.sample_target['points']
        self.target_C         = self.sample_target['C'] # normalization constant
        self.target_name      = self.sample_target['name']

        self.source_landmarks = config['source_landmarks']
        self.target_landmarks = config['target_landmarks']

        self.lands_source = self.source_uvs[self.source_landmarks].float()
        self.lands_target = self.target_uvs[self.target_landmarks].float()
        self.alignment_type = config['alignment_type']
        self.sampling_type = config['sampling_type']
        
        
        bad_vertex = self.source_landmarks[0]
        bad_faces = [i for i in range(self.source_faces.shape[0]) if bad_vertex in self.source_faces[i]]
        self.bad_faces = bad_faces
        
        
           
        ## compute R and t to position the two parametrization (not needed if start randomly)
        if self.alignment_type=="rotation":
            self.R = self.compute_lands_rotation(self.lands_source, self.lands_target)
            self.R = self.R.t()
            
        elif self.alignment_type=="mobius_triplet":
            print('Using alignment type : mobius triplet by inversions.')
            self.R = self.compute_lands_mobius_triplet_inversion(self.lands_source, self.lands_target)
            
        elif self.alignment_type=="lsq_affine":
            print('Using alignment type : least squares mobius (complex affine).')
            self.R = self.compute_lands_lsq_mobius(self.lands_source, self.lands_target)

        elif self.alignment_type=="mobius": ### this is old, we prefer inversion-based mobius (mobius triplet)
            print('Using alignment type : mobius.')
            self.R = self.compute_lands_mobius(self.lands_source, self.lands_target)
            
        elif self.alignment_type=="inversion":
            print('Using alignment type : inversion.')
            self.R = self.compute_lands_inversion(self.lands_source, self.lands_target)
            
        elif self.alignment_type=="rotate2pole":
            print('Using alignment type : rotate2pole.')
            self.R = self.compute_lands_rotate2pole(self.lands_source, self.lands_target)
            
        elif self.alignment_type=="none":
            print('Default: proceeding with no alignment function.')
            self.R = None
        else:
            raise ValueError('No alignment type specified.')
        
        
        ## get icosphere
        sphere = trimesh.load('../data/analytic/sphere/sphere2.obj')
        self.icosphere = dict()
        self.icosphere['vertices'] = torch.tensor(sphere.vertices)
        self.icosphere['faces'] = sphere.faces
        
        ## read GT map if exists
        self.read_map(config)

    def read_map(self, config):
        if 'map_gt' not in config:
            return

        ## untested code, never had a GT map in the experiment
        map = np.loadtxt(config['map_gt'])
        self.map_gt = torch.from_numpy(map).long()

        if self.source_uvs.size(0) != self.map.size(0):
            print('different number of vertices, reverting the mapping')

            self.map_gt = torch.zeros_like(self.map_gt).long()
            self.map_gt[self.sample_source['V_idx_original']] = torch.from_numpy(map).long()



    def __len__(self):
        return 1


    def __getitem__(self, index):

        ## sample 2D parametrization
        params_to_sample = [self.source_uvs]
        
        
        if self.sampling_type == "landmark avoidant":                
            ## print('ldmk avoidant sampling')
            weights = torch.ones((self.source_faces).shape[0])
            weights[self.bad_faces] *= 0.0
            weights *= 1.0/torch.sum(weights)
        
            _, _, params_all = sample_surface(self.num_points, self.source_points,
                                    self.source_faces, params_to_sample, weights = weights, method='pytorch3d')
        
        else:
            _, _, params_all = sample_surface(self.num_points, self.source_points,
                                    self.source_faces, params_to_sample, method='pytorch3d')

        params = params_all[0]
        icosphere = trimesh.load('../data/analytic/sphere/sphere0.obj')
        icosphere_dict = {'vertices': torch.tensor(icosphere.vertices) , 'faces':icosphere.faces }

        data_dict = {
            'source_points':    params,
            'R':                self.R,
            #'t':                -self.t,
            'C_source':         self.source_C,
            'C_target':         self.target_C,
            'target_domain':    None,
            'boundary':         None, #self.source_boundary,
            'landmarks':        self.lands_source,
            'target_landmarks': self.lands_target,
            'icosphere': icosphere_dict
        }

        ## add domain triangulation in case domain is not a disk or a square
        if 'domain_faces' in self.sample_target:
            data_dict['target_domain'] = self.sample_target['domain_vertices'][self.sample_target['domain_faces']]

        return data_dict


    def num_checkpointing_samples(self):
        return 1

    def get_checkpointing_sample(self, index):

        data_dict = {}
        data_dict['source_points']    = self.source_uvs
        data_dict['target_points']    = self.target_uvs
        data_dict['source_faces']     = self.source_faces
        data_dict['target_faces']     = self.target_faces
        data_dict['target_domain']    = None
        data_dict['R']                = self.R
        #data_dict['t']                = -self.t
        data_dict['C_source']         = self.source_C
        data_dict['C_target']         = self.target_C
        data_dict['landmarks']        = self.lands_source
        data_dict['target_landmarks'] = self.lands_target
        data_dict['target_name']      = self.target_name
        data_dict['source_name']      = self.source_name
        #data_dict['boundary']         = self.source_boundary

        data_dict['target_points_3D'] = self.target_points

        ## optional parameters inside the sample for visualization
        if 'visual_v' in self.sample_source:
            data_dict['visual_uv'] = {'xz':self.sample_source['visual_v'][:,[0,2]], # this is okay for cp2
                                      'xy':self.sample_source['visual_v'][:,[0,1]], # this is okay for cp2
                                      'yz':self.sample_source['visual_v'][:,[1,2]] }# this is okay for cp2
        if 'visual_v' in self.sample_target:
            data_dict['visual_uv_target'] = {'xz':self.sample_target['visual_v'][:,[0,2]], # this is okay for cp2
                                      'xy':self.sample_target['visual_v'][:,[0,1]], # this is okay for cp2
                                      'yz':self.sample_target['visual_v'][:,[1,2]] }# this is okay for cp2

        if hasattr(self, 'map'):
            data_dict['map_gt']   = self.map_gt

        if 'domain_faces' in self.sample_target:
            data_dict['target_domain'] = self.sample_target['domain_vertices'][self.sample_target['domain_faces']]

        if 'oversampled_param' in self.sample_source:
            data_dict['oversampled_param'] = self.sample_source['oversampled_param'].float()
            data_dict['oversampled_faces'] = self.sample_source['oversampled_faces'].long()


        return data_dict





class SurfaceMapSingularDataset(DatasetMixin):

    def __init__(self, config):

        self.config     = config
        #print('config',config)
        self.num_points = config['num_points']

        self.sample_source = self.read_sample(config['sample_source'])
        ## extract all data
        self.source_uvs       = self.sample_source['param']
        self.source_faces     = self.sample_source['faces']
        self.source_points    = self.sample_source['points']
        self.source_C         = self.sample_source['C'] # normalization constant
        


        self.source_landmarks = None
        self.target_landmarks = None 

       
        self.R = torch.eye(3)   ###no alignment needed 
        self.source_name      = self.sample_source['name']

        self.sampling_type = config['sampling_type']       
           
               
        
        ## get icosphere
        sphere = trimesh.load('../data/analytic/sphere/sphere2.obj')
        self.icosphere = dict()
        self.icosphere['vertices'] = torch.tensor(sphere.vertices)
        self.icosphere['faces'] = sphere.faces
        
        ## read GT map if exists
        self.read_map(config)

    def read_map(self, config):
        if 'map_gt' not in config:
            return

        ## untested code, never had a GT map in the experiment
        map = np.loadtxt(config['map_gt'])
        self.map_gt = torch.from_numpy(map).long()

        if self.source_uvs.size(0) != self.map.size(0):
            print('different number of vertices, reverting the mapping')

            self.map_gt = torch.zeros_like(self.map_gt).long()
            self.map_gt[self.sample_source['V_idx_original']] = torch.from_numpy(map).long()



    def __len__(self):
        return 1


    def __getitem__(self, index):

        ## sample 2D parametrization
        params_to_sample = [self.source_uvs]
        
        
        _, _, params_all = sample_surface(self.num_points, self.source_points,
                                    self.source_faces, params_to_sample, method='pytorch3d')

        params = params_all[0]
        icosphere = trimesh.load('../data/analytic/sphere/sphere0.obj')
        icosphere_dict = {'vertices': torch.tensor(icosphere.vertices) , 'faces':icosphere.faces }

        data_dict = {
            'source_points':    params,
           
            #'t':                -self.t,
            'C_source':         self.source_C,
              
            'icosphere': icosphere_dict
        }

        ## add domain triangulation in case domain is not a disk or a square
        ###if 'domain_faces' in self.sample_target:
        ###    data_dict['target_domain'] = self.sample_target['domain_vertices'][self.sample_target['domain_faces']]

        return data_dict


    def num_checkpointing_samples(self):
        return 1

    def get_checkpointing_sample(self, index):

        data_dict = {}
        data_dict['source_points']    = self.source_uvs

        data_dict['source_faces']     = self.source_faces

        #data_dict['R']                = self.R
        #data_dict['t']                = -self.t
        data_dict['C_source']         = self.source_C
        data_dict['R']                = self.R
        #data_dict['t']                = -self.t

        data_dict['source_name']      = self.source_name
        #data_dict['boundary']         = self.source_boundary

      
        ## optional parameters inside the sample for visualization
        if 'visual_v' in self.sample_source:
            data_dict['visual_uv'] = {'xz':self.sample_source['visual_v'][:,[0,2]], # this is okay for cp2
                                      'xy':self.sample_source['visual_v'][:,[0,1]], # this is okay for cp2
                                      'yz':self.sample_source['visual_v'][:,[1,2]] }# this is okay for cp2
        
        if hasattr(self, 'map'):
            data_dict['map_gt']   = self.map_gt

        # if 'domain_faces' in self.sample_target:
        #     data_dict['target_domain'] = self.sample_target['domain_vertices'][self.sample_target['domain_faces']]

        if 'oversampled_param' in self.sample_source:
            data_dict['oversampled_param'] = self.sample_source['oversampled_param'].float()
            data_dict['oversampled_faces'] = self.sample_source['oversampled_faces'].long()


        return data_dict
