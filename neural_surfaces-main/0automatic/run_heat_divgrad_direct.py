import numpy as np
import os
import sys
import shutil
#from normalise_mesh import normalise_mesh
import re
import torch
import json

from runners import MainRunner
from mains.experiment_configurator import ExperimentConfigurator
from differential import *

if len(sys.argv)>1:
        name = sys.argv[1]
        #name = sys.argv[2]
else:
        name = input('Input the name of the SNS. E.g. <<MAX10606>> .')

####### move  the intial weights and param to the correct place
#os.system('cp ../data/SNS/'+name+'/param.pth ../data/heat/'+name+'/current_param.pth')
os.system('cp ../data/eigenfunc/HUMAN24461/orthoweights/ortho4.pth ../data/heat/'+name+'/current_field_weights.pth')


#sample = torch.load('../data/heat/'+name+'/current_param.pth')
#print(sample.keys())
#sample['discrete_eigenfunc'] = sample['param']
        

from sample_preparation.eigenfunc import sample_prep_functions

sample = sample_prep_functions.prepare_sample(name, target_number_samples= 50000, initial_number_samples=200000)






torch.save(sample, '../data/heat/'+name+'/current_param.pth')




with open('experiment_configs/heat/' + name + '.json') as f:
    config = json.load(f)


diffmod = DifferentialModule()
modules_creator = ExperimentConfigurator()
surf_runner = MainRunner('experiment_configs/overfit/ARMADILLO21622.json', modules_creator)
field_runner = MainRunner('experiment_configs/eigenfunc/MAX10606.json', modules_creator)   
surf_model = surf_runner.get_model()


max_iter = config['params']['max_iter']
d = config['params']['flow_step_size']

#################### do the meancurv & normals calculation #########################


sample = torch.load('../data/SNS/'+name+'/param.pth')#sample sample dict can be used for both wfield and surface values
surf_weights = torch.load('../data/SNS/'+name+'/weights.pth', map_location=torch.device('cpu'))

surf_model.load_state_dict(surf_weights)
surf_model.eval()

#compute H and n
param = sample['param']
tensorvertices = torch.Tensor(param)
tensorvertices.requires_grad=True
output_vertices = surf_model.forward(tensorvertices)
#normals = diffmod.compute_normals(out = output_vertices, wrt = tensorvertices).detach()
#H,_,_,_,_,_,_ = diffmod.compute_curvature(out = output_vertices, wrt = tensorvertices, compute_principal_directions=False)
#H = H.detach()

#jacobian3D = diffmod.gradient(out=output_vertices, wrt=tensorvertices)
#inv_jacobian3D = torch.linalg.inv(jacobian3D)
#print(inv_jacobian3D.shape)
#transpose_inv_jacobian3D = inv_jacobian3D.transpose(2,1)
#print(transpose_inv_jacobian3D.shape)





os.system('cp ../data/heat/'+name+'/current_field_weights.pth ../data/heat/'+name+'/field_weights')      
#evaluate field values and compute new values
        
field_weights = torch.load('../data/heat/'+name+'/current_field_weights.pth', map_location=torch.device('cpu'))
        
field_model = field_runner.get_model()
field_model.load_state_dict(field_weights)
field_model.train()









for i in range(max_iter):
	

	print(i)

	#os.system('cp ../data/heat/'+name+'/current_field_weights.pth ../data/heat/'+name+'/field_weights'+str(i)+'.pth')
	
	
	#evaluate field values and compute new values
	
	#field_weights = torch.load('../data/heat/'+name+'/current_field_weights.pth', map_location=torch.device('cpu'))
	
	#field_model = field_runner.get_model()
	#field_model.load_state_dict(field_weights)
	#field_model.eval()

	
	
	#tensorvertices.requires_grad=True
	g_values = field_model.forward(tensorvertices).mean(-1)
	
	#print(transpose_inv_jacobian3D.shape)	
	#grad_f = (diffmod.gradient(out = f_values.unsqueeze(-1), wrt=tensorvertices) @ inv_jacobian3D).squeeze()  ## differentiate f wrt the surface
	#grad_h = (transpose_inv_jacobian3D @ diffmod.gradient(out = g_values.unsqueeze(-1), wrt=tensorvertices).squeeze().unsqueeze(-1)   ).squeeze()	

	#print(grad_h.shape)

	#hessian_h = (   transpose_inv_jacobian3D @  diffmod.gradient(out = grad_h.squeeze(), wrt = tensorvertices)  ).detach()
	#grad_h = grad_h.detach()

	lb = diffmod.laplace_beltrami_divgrad(out=output_vertices, wrt=tensorvertices, f = g_values, f_defined_on_sphere=True)
	
	print('lb', lb.max(), lb.min())
	#update sample
	
	#input()
	#sample['discrete_eigenfunc'] = (g_values + d*lb ).detach()
	
	#print('current',(g_values).max(), (g_values).min() )
	#print('target',(g_values+d*lb).max(), (g_values+d*lb).min() )
	
	#torch.save(sample, '../data/heat/'+name+'/current_param.pth')
	
	#print('prepared iteration',i)      
	#print(sample.keys())
	#print(sample['param'].shape, sample['discrete_eigenfunc'].shape)	


	optimizer = torch.optim.SGD(field_model.parameters(), lr=1e-10)
	#12 zeros: nothing happens
        #14 zeros good for max?
	dirichlet = (lb*lb).sum()
	
	dirichlet.backward(retain_graph=True)
	
	optimizer.step()


	del g_values
	del lb
	#del hessian_h 
	#del grad_h
	#print('param shape', sample['discrete_eigenfunc'].shape)
	#overfit to this sample
	#os.system('python -m mains.training experiment_configs/heat/'+name+'.json')	
	torch.save(field_model.state_dict(), "../data/heat/SPIKE999/model_weights_"+str(i)+".pth")
























