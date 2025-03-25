
import torch
from torch.nn import Module
from torch.nn import functional as F
from .mlp import *





class SurfacePathMap(Module):

    def __init__(self, config):
        super().__init__()

        sns_struct = config['sns'] # structure of the surface map
        sphere_path_struct            = config['path'] # structure of the neural map
        

        self.sns = globals()[sns_struct['name']](sns_struct) # create network for sns
        
        self.sns.load_state_dict(torch.load(sns_struct['path'], map_location='cpu'))
        
        
        
        self.sphere_path_map   = globals()[sphere_path_struct['name']](sphere_path_struct) # create network for path
        ## load surface models
        

        ## disable grads for surface map
        self.disable_network_gradient(self.sns)


    def disable_network_gradient(self, network):
        for param in network.parameters():
            param.requires_grad_(False)



    
    
    def forward(self, points1D, tgt_startpoint, tgt_endpoint):
       
        
        mapped_points   = self.forward_map(points1D, tgt_startpoint, tgt_endpoint) # forward path map
        #print(f"mapped_points: {mapped_points}")
        #print(tgt_startpoint, tgt_endpoint)

        points3D = self.sns(mapped_points) # forward target surface map
        
        return points3D, mapped_points
        ### return mapped_points, points3D
        
        '''
        displacements = points1D * (1 - points1D) * self.forward_map(points1D)        
        linear_part = points1D * tgt_endpoint + (1-points1D) *  tgt_startpoint

        mapped_points   = 
        points3D = self.sns(mapped_points) # forward target surface map
        
        return points3D, mapped_points
        '''
    
    
    def forward_map(self, points1D, tgt_startpoint, tgt_endpoint):                                                                                                                                       
        '''
        mapped_points =self.sphere_path_map(interval_points)                                                                                                                                              
        #project_to_sphere
        mapped_points = F.normalize(mapped_points,p=2,dim=1)
        return mapped_points
        '''
        displacements = points1D * (1 - points1D) * self.sphere_path_map(points1D)        
        linear_part = points1D * tgt_endpoint + (1-points1D) *  tgt_startpoint

        mapped_points = F.normalize(displacements + linear_part, p=2,dim=1)
        #print(mapped_points,'mapped points') 
        #points3D = self.sns(mapped_points) # forward target surface map

        return mapped_points 
