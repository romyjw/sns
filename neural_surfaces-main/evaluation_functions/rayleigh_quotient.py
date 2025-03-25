from differential import DifferentialModule
from runners import TrainRunner
import numpy as np
import torch

from runners import MainRunner
from mains.experiment_configurator import ExperimentConfigurator

from utils.rejection_sampling import rejection_sampling
from utilsR import batches_diff_quant

import sys
initial_number_samples = 2000000
target_number_samples = 100000
batch_size = 20000


surf_name = 'icosphere5'
eigenfunc_num = 5

surf_weights = torch.load('../data/SNS/'+surf_name+'/weights.pth', map_location = torch.device('mps'))
field_weights = torch.load('../data/eigenfunc/'+surf_name+'/orthoweights/ortho'+str(eigenfunc_num)+'.pth', map_location = torch.device('mps'))


diffmod = DifferentialModule()

modules_creator = ExperimentConfigurator()


field_runner = MainRunner('experiments/eigenfunc/test.json', modules_creator)
field_model = field_runner.get_model()

field_model.load_state_dict(field_weights)
field_model.eval()

surf_runner = MainRunner('experiments/overfit/ARMADILLO21622.json', modules_creator)
surf_model = surf_runner.get_model()

surf_model.load_state_dict(surf_weights)
surf_model.eval()



gaussian_samples = torch.randn(initial_number_samples,3)
sphere_samples = (gaussian_samples.T / torch.sqrt(gaussian_samples.pow(2).sum(-1)) ).T


big_surface_batches_diff_quant = batches_diff_quant.batches_diff_quant(sphere_samples, surf_model, diffmod, batch_size)
big_surface_batches_diff_quant.compute_area_distortions()

area_density = big_surface_batches_diff_quant.all_area_distortions

sample_points, area_density, keeping_iis = rejection_sampling(torch.Tensor(sphere_samples), torch.Tensor(area_density), target_number_samples) 
		
param = sphere_samples[keeping_iis, :]
		




def dirichlet_energy(param, field_model, surf_model):
        #print ('TIJ shape in dirichlet energy function',transpose_inverse_jacobians.shape)
        
        param.requires_grad = True

        points = surf_model.forward(param)
        
        jacobians = diffmod.gradient(out = points, wrt = param).detach()
        transpose_inverse_jacobians = torch.linalg.inv(jacobians).transpose(2,1)
        
        print('TIJ', transpose_inverse_jacobians.shape)
        
        normals = diffmod.compute_normals(out=points, wrt=param).detach()
        
        
        
        
        f_values = field_model.forward(param).mean(-1).unsqueeze(-1)
        print('f vals', f_values.shape)
        
        DfDsph = diffmod.gradient(out = f_values, wrt = param).squeeze().unsqueeze(-1) # it's 4048 x 1 x 3 if I don't squeeze. 
        print('DfDsph', DfDsph.shape)

        DfDsurf =  ( transpose_inverse_jacobians @ DfDsph).squeeze()  #When the surface is just the sphere, the transpose inverse jacobians are all 3x3 identity matrices.
        #DfDsurf = DfDsph.squeeze() 
                
        coeffs = (DfDsurf*normals).sum(-1)
        covariant_grad = DfDsurf - (coeffs *  normals.T).T

        #dA = 4.0*np.pi/f_values.shape[0]
        #dirichlet_energy = (covariant_grad @ covariant_grad.T).sum()*dA
        
        #dA = 4.0*np.pi/f_values.shape[0]
        dirichlet_energy = (covariant_grad.pow(2).sum(-1)).mean()*4.0*np.pi
        
        #loss = dirichlet_energy #+ norm_regularisation
        return dirichlet_energy, f_values, param
        
        
def rayleigh_quotient(param, field_model, surf_model ):
        DE, f_values, param = dirichlet_energy(param, field_model, surf_model)
        l2norm_sq = ( 4.0*np.pi*(f_values.pow(2)).mean() )
        
        return DE/l2norm_sq




rayleigh_quotient = rayleigh_quotient(param, field_model, surf_model)

print('RQ with '+str(param.shape[0])+' points is '+ str(rayleigh_quotient))











