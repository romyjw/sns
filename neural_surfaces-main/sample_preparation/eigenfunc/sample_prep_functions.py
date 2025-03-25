
import torch
from differential import *
from differential import batches_diff_quant
from runners import MainRunner
from mains.experiment_configurator import ExperimentConfigurator
from utils.rejection_sampling import rejection_sampling

from utils.mesh_LB import L_cotan, L_uniform, laplace_beltrami_cotan_MC
from scipy import sparse
import trimesh
import numpy as np




def prepare_sample(SNS_name, target_number_samples=10000, initial_number_samples=100000):

	
    ### generate samples on a sphere
    ### compute output points, FFF, normals, jacobians, density, using the neural surface
    ### reject points so that sampling becomes uniform on surface
    ### return sample including points, normals, param, jacobians
    
    ### if the shape is just the sphere, the normals, density, jacobians, ouput points are all trivial.
    
    
    sample = {}
    
    gaussian_samples = torch.randn(initial_number_samples,3)
    sphere_samples = (gaussian_samples.T / torch.sqrt(gaussian_samples.pow(2).sum(-1)) ).T
	
    diffmod = DifferentialModule() #we need a differential module for computing normals and area density and jacobians

   
    ######### set up model and runner etc. for the spherical neural surface ##########
    modules_creator = ExperimentConfigurator()
    runner = MainRunner('experiment_configs/overfit/ARMADILLO21622.json', modules_creator)
    model = runner.get_model()
    



    weights = torch.load('../data/SNS/'+SNS_name+'/weights.pth', map_location=torch.device('cpu'))

    model.load_state_dict(weights)
    model.eval()

    my_batches_diff_quant = batches_diff_quant.batches_diff_quant(gaussian_samples, model, diffmod, 10000) 


	
    ######################################################################################
    param = sphere_samples
    param.requires_grad = True
    points = model.forward(param) ##########put through forward model
    
    
    my_batches_diff_quant.compute_area_distortions()
    area_density = my_batches_diff_quant.all_area_distortions

    #area_density = diffmod.compute_area_distortion(out = points, wrt = sphere_samples)
     	
    #visualise_pointcloud(vertices, stretch_cmap(area_density)) ########## visualise pointcloud before rejections
    	
    sphere_samples, area_density, keeping_iis = rejection_sampling(sphere_samples, area_density, target_number_samples) ######################## reject points
            
    normals = diffmod.compute_normals(out=points, wrt=param)[keeping_iis,:]
    jacobians3x3 = diffmod.gradient(out=points, wrt=param)[keeping_iis,:,:] ### would be more efficient to get this from the normals calculations
    transpose_inverse_jacobians = torch.linalg.inv(jacobians3x3).transpose(1,2)
    param = sphere_samples
    points = points[keeping_iis,:]


    sample['param']      = param.detach()
    
    sample['points'] = points.detach()
    sample['normals']        = normals.detach()
    sample['transpose_inverse_jacobians'] = transpose_inverse_jacobians.detach()
        
    sample['name']           = SNS_name
    sample['ortho_functions'] = None

    return sample
    

def prepare_overfit_sample(SNS_name, eigNumber):

	
    ### generate samples on a sphere
    ### compute output points, FFF, normals, jacobians, density, using the neural surface
    ### reject points so that sampling becomes uniform on surface
    ### return sample including points, normals, param, jacobians
    
    ### if the shape is just the sphere, the normals, density, jacobians, ouput points are all trivial.
        
    ######### set up model and runner etc. for the spherical neural surface ##########
    modules_creator = ExperimentConfigurator()
    runner = MainRunner('experiments/overfit/ARMADILLO21622.json', modules_creator)
    model = runner.get_model()
    weights = torch.load('../data/SNS/'+SNS_name+'/weights.pth', map_location=torch.device('cpu'))

    model.load_state_dict(weights)
    model.eval()
    
    
    sphere_tm = trimesh.load('../data/analytic/sphere/sphere5.obj')
    vertices = torch.Tensor(sphere_tm.vertices)
    param = vertices
    
    param.requires_grad = True
    points = model.forward(param) ##########put through forward model
    
    surf_tm = sphere_tm.copy()
    surf_tm.vertices = np.array(points.detach())

    sample = {} 
    sample['name'] = SNS_name
    sample['param']      = param.detach()
    sample['points']         = points.detach()
	
    ################ compute LB operator and eigenvectors in order to get discrete functions to overfit to
    
    
    L,M,Minv,C = L_cotan(surf_tm)
    sparse_L_cotan_matrix = Minv @ C
    L_cotan_matrix = sparse_L_cotan_matrix.todense().T
	
    vals, vecs = sparse.linalg.eigs(sparse_L_cotan_matrix, k=5, which='SM')
    
    area = 4.0*np.pi
    vecs = vecs/((area*(vecs**2)).mean(0)**0.5)
    print('discrete eigenvalue is: ',vals[eigNumber])
    print('discrete eigenfunction is: ', vecs[:, eigNumber])
    sample['discrete_eigenfunc'] = torch.Tensor(vecs[:,eigNumber])
    
    sample['C'] = 1

    return sample



