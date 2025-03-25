
import torch
from torch.nn import Module

from .mlp import *
from geometric_functions import *


def robust_acos(x):
    return torch.acos(torch.clamp(x, -1.0, 1.0))


def spherical_polar(points3D):
    if points3D.isnan().any():
        raise ValueError('I am raising error because spherical polar had a bad input')
    
    
    u = robust_acos(points3D[:,2])
    
    v=torch.atan2(points3D[:,1], points3D[:,0])
    polar = torch.stack((u,v)).T
    

    if u.isnan().any():
        raise ValueError('I am raising error because u  calc resulted in Nans')
    if v.isnan().any():
        raise ValueError('I am raising error because v  calc resulted in Nans')

    if polar.isnan().any():
        raise ValueError('I am raising error because spherical polar calc resulted in Nans')
    return polar

def inv_spherical_polar(points2D):
    ######### sin u * cos v,    sin u * sin v,       cos u
    u = points2D[:,0]
    v = points2D[:,1]
    X = torch.sin(u)*torch.cos(v)
    Y = torch.sin(u)*torch.sin(v)
    Z = torch.cos(u)
    
    cartesian = torch.stack((X,Y,Z)).T
    
    return cartesian


def R(batched_complex):
	real = torch.zeros((batched_complex.shape[0], 2), device = batched_complex.device)
	real[:,0] = torch.real(batched_complex)
	real[:,1] = torch.imag(batched_complex)
	return real

def C(batched_2D):
	return batched_2D[:,0] + 1.0j*(batched_2D[:,1])







class ParametrizationMap(Module):

    def __init__(self, config):
        super().__init__()

        source_surface_struct = config['source_surface'] # structure of the surface map
        map_struct            = config['map'] # structure of the neural map
        self.alignment_type = config['alignment_type']

        self.source_surface = globals()[source_surface_struct['name']](source_surface_struct) # create network for surface map
        self.neural_map     = globals()[map_struct['name']](map_struct) # create network for neural map

        ## load surface models
        if 'path' in map_struct:
            self.neural_map.load_state_dict(torch.load(map_struct['path'], map_location='cpu'))

        self.source_surface.load_state_dict(torch.load(source_surface_struct['path'], map_location='cpu'))

        ## disable grads for surface map
        self.disable_network_gradient(self.source_surface)


    def disable_network_gradient(self, network):
        for param in network.parameters():
            param.requires_grad_(False)


    def forward(self, points2D):
        
        
        points3D_init = self.source_surface(points2D) # forward surface map
        mapped_points   = self.neural_map(points2D) # forward neural map
        points3D_final = self.source_surface(mapped_points)

        return mapped_points, points3D_init, points3D_final

class NeuralMapSingular(ParametrizationMap):

    def __init__(self, config):
        super().__init__(config)

        
        
                
        ##initialise weights
        self.init_map_weights()

        ## disable grads
        #self.init_map_weights()
        self.disable_network_gradient(self.source_surface)
        
    
    def forward(self, points2D, alignment_sig=None):
        
        mapped_points   = self.forward_map(points2D) # forward neural map
        points3D = self.source_surface(mapped_points) # forward target surface map
        
        return points3D, mapped_points, None
        ### return mapped_points, points3D
    
    
    def init_map_weights(self):

        #print('init map weights called')  
    # init with identity by adding point at the end                                                                                                                                    
        with torch.no_grad():                                                                                                                                                               
            if hasattr(self.neural_map.mlp[-1], 'weight'):                                                                                                                                
                self.neural_map.mlp[-1].weight.fill_(0.0001)      
                #if hasattr(self.neural_map.mlp[-1], 'bias'):                                                                                                                             
                #    self.neural_map.mlp[-1].bias.fill_(0.0)                                                                                                                                    
            else:                                                                                                                                                                               
                for i in range(len(self.neural_map.mlp)):                                                                                                                                  
                    self.neural_map.mlp[i].residual[2].weight *= 0.01                                                                                                                       
                    #self.neural_map.mlp[i].residual[2].bias *= 0.01  
                
    def forward_map(self,sphere_points, alignment_sig=None, no_normalise=True):                                                                                                                                               

       
                                                                                                                                                              
                 
        mapped_points = sphere_points[..., :3] + self.neural_map(sphere_points)                                                                                                                                              
        
        if no_normalise==False:
            mapped_points = F.normalize(mapped_points,p=2,dim=1)                                                                                                                    

        return mapped_points    



class NeuralMap(ParametrizationMap):

    def __init__(self, config):
        super().__init__(config)

        target_surface_struct = config['target_surface'] # get structure of target surface map

        self.target_surface = globals()[target_surface_struct['name']](target_surface_struct) # create network for target surface map

        ## load surface models
        self.target_surface.load_state_dict(torch.load(target_surface_struct['path'], map_location='cpu'))
        
        ##initialise weights
        self.init_map_weights()

        ## disable grads
        #self.init_map_weights()
        self.disable_network_gradient(self.target_surface)
        
    
    def forward(self, points2D, alignment_sig=None):
        
        points3D_source = self.source_surface(points2D) # forward source surface map
        mapped_points   = self.forward_map(points2D, alignment_sig) # forward neural map
        points3D_target = self.target_surface(mapped_points) # forward target surface map

        return points3D_target, mapped_points, points3D_source
    
    
    def init_map_weights(self):

        #print('init map weights called')  
    # init with identity by adding point at the end                                                                                                                                    
        with torch.no_grad():                                                                                                                                                    
            if hasattr(self.neural_map.mlp[-1], 'weight'):                                                                                                                                
                self.neural_map.mlp[-1].weight.fill_(0.0001)      
                #if hasattr(self.neural_map.mlp[-1], 'bias'):                                                                                                                             
                #    self.neural_map.mlp[-1].bias.fill_(0.0)                                                                                                                                    
            else:                                                                                                                                                                               
                for i in range(len(self.neural_map.mlp)):                                                                                                                                  
                    self.neural_map.mlp[i].residual[2].weight *= 0.01                                                                                                                       
                    #self.neural_map.mlp[i].residual[2].bias *= 0.01  
                
    def forward_map(self,sphere_points, alignment_sig=None, no_normalise=True):                                                                                                                                               

        ## print(self.alignment_type)
        if self.alignment_type=='rotation':
            aligned_points = sphere_points.matmul(alignment_sig)
        elif self.alignment_type=='mobius_triplet':
            aligned_points = full_mobius_transform(sphere_points, alignment_sig) 
        elif self.alignment_type=='lsq_affine':
            aligned_points = full_mobius_transform(sphere_points, alignment_sig)   
        elif self.alignment_type=='none':
            aligned_points = sphere_points
        elif self.alignment_type=='inversion':
            aligned_points = invert_sphere1(sphere_points, alignment_sig[0])
        elif self.alignment_type=='rotate2pole':
            aligned_points = rotate2pole1(sphere_points, alignment_sig[0])
        else:
            raise ValueError('No alignment type was specified.')

                                                                                                                                                              
        #if R is not None:      
        #    if torch.cuda.is_available():
        #        try:                                                                                                                                                            
        #            rot_points = points2D.matmul(R.to(0))
        #        except:
        #            rot_points = points2D.matmul(R[0].to(0))
        #    else:
        #        try:                                                                                                                                                            
        #            rot_points = points2D.matmul(R)
        #        except:
        #            rot_points = points2D.matmul(R[0])
         
        mapped_points = aligned_points[..., :3] + self.neural_map(aligned_points)                                                                                                                                              
        
        if no_normalise==False:
            mapped_points = F.normalize(mapped_points,p=2,dim=1)                                                                                                             
        

        if self.alignment_type=='rotation' or self.alignment_type=='mobius_triplet' or self.alignment_type=='lsq_affine':
            aligned_mapped_points = mapped_points
        elif self.alignment_type=='none':
            aligned_mapped_points = mapped_points
        elif self.alignment_type=='inversion':
            aligned_mapped_points = invert_sphere2(mapped_points, alignment_sig[1])
        elif self.alignment_type=='rotate2pole':
            aligned_mapped_points = rotate2pole2(mapped_points, alignment_sig[0])
        else:
            raise ValueError('No alignment type was specified.')

        

        return aligned_mapped_points    


class NeuralMapAngular(ParametrizationMap):

    def __init__(self, config):
        super().__init__(config)

        target_surface_struct = config['target_surface'] # get structure of target surface map

        self.target_surface = globals()[target_surface_struct['name']](target_surface_struct) # create network for target surface map

        ## load surface models
        self.target_surface.load_state_dict(torch.load(target_surface_struct['path'], map_location='cpu'))
        
        ##initialise weights
        self.init_map_weights()

        ## disable grads
        #self.init_map_weights()
        self.disable_network_gradient(self.target_surface)
        
    
    def forward(self, spherepoints, alignment_sig, learning=True):
        points3D_source = self.source_surface(spherepoints) # forward source surface map
        mapped_sphere_points = self.forward_map(spherepoints, alignment_sig, learning)
        
        points3D_target = self.target_surface(mapped_sphere_points)#


        return points3D_target, mapped_sphere_points, points3D_source
    
    
    def init_map_weights(self):

        #print('init map weights called')  
    # init with identity by adding point at the end                                                                                                                                    
        with torch.no_grad():                                                                                                                                                               
            if hasattr(self.neural_map.mlp[-1], 'weight'):                                                                                                                                
                self.neural_map.mlp[-1].weight.fill_(0.0001)      
                #if hasattr(self.neural_map.mlp[-1], 'bias'):                                                                                                                             
                #    self.neural_map.mlp[-1].bias.fill_(0.0)                                                                                                                                    
            else:                                                                                                                                                                               
                for i in range(len(self.neural_map.mlp)):                                                                                                                                  
                    self.neural_map.mlp[i].residual[2].weight *= 0.01                                                                                                                       
                    #self.neural_map.mlp[i].residual[2].bias *= 0.01  
                
    def forward_map(self, sphere_points, alignment_sig=None, learning = True):
    
        if self.alignment_type=="inversion":
        	#print('The alignment sig is ', alignment_sig)
        	aligned_points = invert_sphere1(sphere_points, alignment_sig[0])
        elif self.alignment_type=="none":
        	aligned_points = sphere_points
        else:
        	raise ValueError('Please specify a valid alignment type in the experiment json file.')
        
        neural_output = self.neural_map(aligned_points)
        traj = neural_output
        u = spherical_polar(aligned_points)[:,0]
        
        
        ###############################
        
        #traj = torch.arctan(traj)/6.0 ###### restrict the angular change to be between -pi/12 and pi/12
        #traj[:,1] *= 0.0 #### fix longitude change at 0
        ###############################
        
        traj[:,0] *=torch.sin(u)

        #c = 0.7
        #traj[:,0] *= torch.nn.functional.relu(1.0 - (c*u - c*torch.pi/2)**2)
        
        if learning==False:#### turn off the learned part of the map, just for testing
        	traj*=0.0 
        
        
        
        
        ########################
        ######## crazy for testing #########
        #traj *=0.0 #######no movement
        #traj[:,1] += 1.0*torch.sin(u)
        #traj[:, 1] = 0.0*traj[:,0] - 0.0*torch.pi
        ########################################
        ########################################
        
        mapped_points3D = inv_spherical_polar(spherical_polar(aligned_points) + traj)
        
        if self.alignment_type=="inversion":
        	mapped_points = invert_sphere2(mapped_points3D, alignment_sig[1])
        elif self.alignment_type=="none":
        	mapped_points = mapped_points3D
        else:
        	raise ValueError('Please specify a valid alignment type in the experiment json file.')
        
        return mapped_points
        



class NeuralMap3DAngular(ParametrizationMap):

    def __init__(self, config):
        super().__init__(config)

        target_surface_struct = config['target_surface'] # get structure of target surface map

        self.target_surface = globals()[target_surface_struct['name']](target_surface_struct) # create network for target surface map

        ## load surface models
        self.target_surface.load_state_dict(torch.load(target_surface_struct['path'], map_location='cpu'))
        
        ##initialise weights
        self.init_map_weights()

        ## disable grads
        #self.init_map_weights()
        self.disable_network_gradient(self.target_surface)
        
    
    def forward(self, spherepoints, alignment_sig):
        points3D_source = self.source_surface(spherepoints) # forward source surface map
        mapped_sphere_points = self.forward_map(spherepoints, alignment_sig)
        points3D_target = self.target_surface(mapped_sphere_points)#


        return points3D_target, mapped_sphere_points, points3D_source
    
    
    def init_map_weights(self):

        #print('init map weights called')  
    # init with identity by adding point at the end                                                                                                                                    
        with torch.no_grad():                                                                                                                                                               
            if hasattr(self.neural_map.mlp[-1], 'weight'):                                                                                                                                
                self.neural_map.mlp[-1].weight.fill_(0.01)      
                #if hasattr(self.neural_map.mlp[-1], 'bias'):                                                                                                                             
                #    self.neural_map.mlp[-1].bias.fill_(0.0)                                                                                                                                    
            else:                                                                                                                                                                               
                for i in range(len(self.neural_map.mlp)):                                                                                                                                  
                    self.neural_map.mlp[i].residual[2].weight *= 0.01                                                                                                                       
                    #self.neural_map.mlp[i].residual[2].bias *= 0.01  
                
    def forward_map(self, sphere_points, alignment_sig=None, flat=False):
           
        #print('The alignment sig is ', alignment_sig)

        ################## DO initial transform ###################
        if self.alignment_type=="mobius":
        	aligned_points = sphere_mobius(mobius_sig[0] ,sphere_points)  
        elif self.alignment_type=="inversion":
        	aligned_points = invert_sphere1(sphere_points, alignment_sig[0])
        else:
        	aligned_points = sphere_points
        	                                                                                                                                                         
        ##################### ############################################                                                                    
                                                                                                                                                    

	
	################## Tidy this up!!! ###############################
        if aligned_points.isnan().any():
            raise ValueError('the input to the neural map contains nans')
                                                                                                                                                       
        neural_output = self.neural_map(aligned_points)  #####+   rot_points[..., :3]
        if neural_output.isnan().any():
            #print('the points', neural_output)
            #print(self.neural_map) 
            raise ValueError('the neural map outputted nans')                                                                                                             
        
        

        if flat==True:
            raise ValueError('called flat fwd map')
            return  neural_output
            
        else:
            angles = neural_output
            # traj[:,0] *= torch.sin(spherical_polar(sphere_points))[:,0]
            #u = spherical_polar(sphere_points)[:,0]
            #c = 0.7
            #traj[:,0] *= torch.nn.functional.relu(1.0 - (c*u - c*torch.pi/2)**2)
            
            alpha, beta, gamma = angles[:, 0], angles[:, 1], angles[:, 2]
            
            I = torch.ones_like(alpha).to(alpha.device)
            zero = torch.zeros_like(I).to(alpha.device)
            
            Rx = torch.cat([I, zero, zero, zero, torch.cos(alpha), -torch.sin(alpha), zero, torch.sin(alpha), torch.cos(alpha)]).reshape((-1, 3, 3))            
            Ry = torch.cat([ torch.cos(beta), zero, -torch.sin(beta),  torch.sin(beta), zero, torch.cos(beta), zero, I, zero]).reshape((-1, 3, 3))
            Rz = torch.cat([ -torch.sin(gamma), torch.cos(gamma), zero, torch.cos(gamma), torch.sin(gamma), zero, zero, zero, I]).reshape((-1, 3, 3))
            			
            				
            #Rx = torch.tensor([[1.0, 0.0, 0.0 ],
            #			[0.0, torch.cos(alpha), -torch.sin(alpha)],
            #			[0.0, torch.sin(alpha), torch.cos(alpha)]])
            
            #Ry = torch.tensor([[torch.cos(beta), 0.0,  -torch.sin(beta)],
            #			[0.0, 1.0, 0.0]
            #			[torch.sin(beta), 0.0, torch.cos(beta)]])
            #
            #Rz = torch.tensor([[ -torch.sin(gamma), torch.cos(gamma), 0.0],
            #			[torch.cos(gamma), torch.sin(gamma), 0.0],
            #			[0.0, 0.0, 1.0]])
            
            R_temp = torch.einsum('bik, bkj -> bij', Rx, Ry)
            R = torch.einsum('bik, bkj -> bij', R_temp, Rz)
            
            mapped_points3D = torch.einsum('bi, bij -> bj', sphere_points, R)
            

            if mapped_points3D.isnan().any():
                raise ValueError('mapped points3D was bad after inv spherical polar.')
            if self.alignment_type=="mobius":
                mapped_points = sphere_mobius(alignment_sig[1] ,mapped_points3D)  ###### poles to target landmarks
            elif self.alignment_type=="inversion":
            	mapped_points = invert_sphere2(mapped_points3D, alignment_sig[1])  ###### poles to target landmarks
            else:
                mapped_points = mapped_points3D
           
            return mapped_points



class NeuralMapStereo(ParametrizationMap):

    def __init__(self, config):
        super().__init__(config)

        target_surface_struct = config['target_surface'] # get structure of target surface map

        self.target_surface = globals()[target_surface_struct['name']](target_surface_struct) # create network for target surface map

        ## load surface models
        self.target_surface.load_state_dict(torch.load(target_surface_struct['path'], map_location='cpu'))
        
        ##initialise weights
        self.init_map_weights()

        ## disable grads
        #self.init_map_weights()
        self.disable_network_gradient(self.target_surface)
        
    
    def forward(self, spherepoints, alignment_sig=None, learning=True):
        
        
        points3D_source = self.source_surface(spherepoints) # forward source surface map (sphere to source surface)
        
        map_output = self.forward_map(spherepoints, alignment_sig, learning)

        mapped_sphere_points = map_output
      
        points3D_target = self.target_surface(mapped_sphere_points) # forward target surface map (mapped sphere to target surface)

        return points3D_target, mapped_sphere_points, points3D_source
    
    
    def init_map_weights(self):

        #print('init map weights called')  
    # init with identity by adding point at the end                                                                                                                                    
        with torch.no_grad():                                                                                                                                                               
            if hasattr(self.neural_map.mlp[-1], 'weight'):                                                                                                                                
                self.neural_map.mlp[-1].weight.fill_(0.0001)      
                #if hasattr(self.neural_map.mlp[-1], 'bias'):                                                                                                                             
                #    self.neural_map.mlp[-1].bias.fill_(0.0)                                                                                                                                    
            else:                                                                                                                                                                               
                for i in range(len(self.neural_map.mlp)):                                                                                                                                  
                    self.neural_map.mlp[i].residual[2].weight *= 0.01                                                                                                                       
                    #self.neural_map.mlp[i].residual[2].bias *= 0.01  
                
    def forward_map(self, sphere_points, alignment_sig=None, learning=True):                                                                                                                                           
       
        ################## DO initial transform ###################
        if self.alignment_type=="inversion":
            aligned_sphere_points = invert_sphere1(sphere_points, alignment_sig[0])  ###### poles to target landmarks
        	
        elif self.alignment_type=='none':
            aligned_sphere_points = sphere_points
        elif self.alignment_type=='rotate2pole':
            aligned_sphere_points = rotate2pole1(sphere_points, alignment_sig[0])
            #print('algned points shape', aligned_sphere_points.shape)
        elif self.alignment_type =='rotation':
            aligned_sphere_points = sphere_points.matmul(alignment_sig)
        else:
            raise ValueError("no alignment type specified. Need to put 'none' if there is none.")
                                                                                                                                                     
        ##################### ############################################                                                                    
        
        flat_points = stereographic2(aligned_sphere_points)                                                                                                                                            
        #print(flat_points.shape, 'is the flat points shape')

        if learning==True:                                                                                                                                               
            mapped_points2D = self.neural_map(flat_points)  +   flat_points
            
            
            ### output = C(self.neural_map(flat_points))
            ### mag = torch.sqrt(flat_points[:,0]**2 + flat_points[:,1]**2)
            
            ### mapped_points2D = R (   C(flat_points)  *   (1.0 + torch.exp(-mag)  * output       )         )
            
        else:
            mapped_points2D = flat_points
        if mapped_points2D.isnan().any():
            print('the points',mapped_points2D)
            print(self.neural_map) 
            raise ValueError('the neural map outputted nans')                                                                                                             
        
        
        
        
        mapped_points3D = stereographic_inv2(mapped_points2D)
        
        #print('mapped pts 3D shape', mapped_points3D.shape)

        #################### final alignment ################
        if self.alignment_type=="inversion":
            mapped_sphere_points = invert_sphere2(mapped_points3D, alignment_sig[1])  ###### poles to target landmarks
        	
        elif self.alignment_type=='none':
            mapped_sphere_points = mapped_points3D
        elif self.alignment_type=='rotate2pole':
            #print('alignment sig 2', alignment_sig[1])       
            mapped_sphere_points = rotate2pole2(mapped_points3D, alignment_sig[1])  
        elif self.alignment_type=='rotation':
            mapped_sphere_points = mapped_points3D   
        else:
            raise ValueError("no alignment type specified. Need to put 'none' if there is none.")
        ###################################################
        #print('mapped sphere points shape', mapped_sphere_points.shape)

        return mapped_sphere_points
   
