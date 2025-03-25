import sys
from runners import MainRunner
from mains.experiment_configurator import ExperimentConfigurator
import torch

import numpy as np
import trimesh#to load mesh
import igl

import pyrender

from .helpers.visualisation_functions import *
#from .subdiv import *
from torch import autograd as Grad

from differential import *

#import rd_helper

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

import re
from utils.formula_converter import *
from utils.custom_ply_writing import *

from differential import batches_diff_quant
from visuals.helpers.colourmappings import *


from torch.nn import functional as F
from SNS_applications.discrete_laplace.mesh_LB import L_cotan, L_uniform, laplace_beltrami_cotan_MC

func_type = 'fancy_sine'



#torch.manual_seed(0)
torch.manual_seed(1)

cmap_name = 'Spectral'#'PRGn'#'PiYG'#'PuOr'#'PRGn'#'Spectral'
cmap = plt.get_cmap(cmap_name)



curv_cmap = plt.get_cmap(cmap_name)
dist_cmap = plt.get_cmap(cmap_name)

error_cmap = plt.get_cmap('Reds')


error_scaling = positive_only_linear1


#from laplace.LBv2 import make_functions ### random functions, etc.
#from laplace.mesh_LB import L_meshLB, L_uniform, laplace_beltrami_meshLB_MC




def make_functions_v2(fieldstyle='sine', derivs=True, num_directions=5, freqlist=[10], scale=10):
	if fieldstyle=='sine':
		''' Function to create lists of lambda functions that define scalar fields in R3. Can also return analytic 1st and 2nd derivatives.'''
		torch.manual_seed(1)
		directions = F.normalize( torch.randn(num_directions,3) , 1 )
		functions = [  lambda x, freq = freq,scale=scale, normal=directions[i,:]:  scale*torch.sin(   freq *	(x * normal ).sum(-1)  ) for freq in freqlist for i in range(num_directions) ]
	
		if derivs==False:
			return functions
		else:
			grads = [ lambda x, freq = freq, scale=scale, normal=directions[i,:]: scale * normal * (freq * torch.cos(freq* (x *normal).sum(-1).squeeze()           ).unsqueeze(-1)) for freq in freqlist for i in range(num_directions) ]
		
			hessians = [ lambda x, freq = freq,scale=scale, normal=directions[i,:]:  scale*(normal.unsqueeze(-1) @ normal.unsqueeze(0)) * (-1 * (freq**2) * torch.sin(freq * (x *normal).sum(-1) ) .unsqueeze(-1).unsqueeze(-1)) for freq in freqlist for i in range(num_directions) ]
			
			return functions, grads, hessians
	
	
	if fieldstyle=='fancy_sine':
		''' Function to create lists of lambda functions that define scalar fields in R3. Can also return analytic 1st and 2nd derivatives.'''
		torch.manual_seed(500)
		directions = F.normalize( torch.randn(num_directions,3) , 1 )
		
		
		def x_dot_n(x,n):
			return ( x * n ).sum(-1)
		
		def middle_part(freq,x,n):
			return freq * 3* (x_dot_n(x,n)**2 +0.5 ) 
		
		
		 
		
		functions = [  lambda x, freq = freq,scale=scale, n=directions[i,:]:  scale*torch.sin(  middle_part(freq,x,n)  )  for freq in freqlist for i in range(num_directions) ]
	
		if derivs==False:
			return functions
		else:
			grads = [ lambda x, freq = freq, scale=scale, n=directions[i,:]: scale * n * (freq * torch.cos(middle_part(freq,x,n)).squeeze() * 6 * x_dot_n(x,n)           ).unsqueeze(-1) for freq in freqlist for i in range(num_directions) ]
			
					
			hessians = [ lambda x, freq = freq,scale=scale, n=directions[i,:]:  6 * scale * freq * (n.unsqueeze(-1) @ n.unsqueeze(0)) * (    torch.cos(middle_part(freq,x,n)) - 6 * (x_dot_n(x,n)**2) * torch.sin(middle_part(freq,x,n)) * freq              ) .unsqueeze(-1).unsqueeze(-1) for freq in freqlist for i in range(num_directions) ]
			
			return functions, grads, hessians
	
	
	
	if fieldstyle=='fancy_sine2':
		''' Function to create lists of lambda functions that define scalar fields in R3. Can also return analytic 1st and 2nd derivatives.'''
		torch.manual_seed(500)
		directions = F.normalize( torch.randn(num_directions,3) , 1 )
		functions = [  lambda x, freq = freq,scale=scale, normal=directions[i,:]:  scale*torch.sin(   freq *	(1/(0.7+(x * normal ).sum(-1)))  ) for freq in freqlist for i in range(num_directions) ]
	
		if derivs==False:
			return functions
		else:
			grads = [ lambda x, freq = freq, scale=scale, normal=directions[i,:]: scale * normal * (freq * torch.cos(freq* (x *normal).sum(-1).squeeze()           ).unsqueeze(-1)) for freq in freqlist for i in range(num_directions) ]
		
			hessians = [ lambda x, freq = freq,scale=scale, normal=directions[i,:]:  scale*(normal.unsqueeze(-1) @ normal.unsqueeze(0)) * (-1 * (freq**2) * torch.sin(freq * (x *normal).sum(-1) ) .unsqueeze(-1).unsqueeze(-1)) for freq in freqlist for i in range(num_directions) ]
			
			return functions, grads, hessians
	
	
	
	
	if fieldstyle=='y02':
		functions = [lambda x : 3*(z.pow(2)) - 1.0]
		
		grads = [lambda x : 6 * z]








#################################################
#################################################

diffmod = DifferentialModule()

## use this in most cases
modules_creator = ExperimentConfigurator()
runner = MainRunner('experiment_configs/overfit/ARMADILLO21622.json', modules_creator)## easier than sys.argv[1]
model = runner.get_model()

surf_name = sys.argv[-3]
weights = torch.load('../data/SNS/'+surf_name+'/weights.pth', map_location=torch.device('cpu'))


level = sys.argv[-2]### sphere level, if using subdivided sphere
batch_size = 2000


show = sys.argv[-1]
##################################################################################

model.load_state_dict(weights)
model.eval()


####################### load a sphere mesh #################


tm_sphere = trimesh.load('../data/analytic/sphere/sphere'+str(level)+'.obj')

	
############################################################



vertices = tm_sphere.vertices
faces = tm_sphere.faces
tensorvertices = torch.Tensor(vertices)


#def get_function_derivatives(vertices, model, diffmod, batch_size, func_type='fancy_sine'):
	

batches_diff_quant = batches_diff_quant.batches_diff_quant(vertices, model, diffmod, batch_size)
batches_diff_quant.compute_SNS()
############# get the info from the batches computation #######
all_output_vertices = batches_diff_quant.all_output_vertices

translation = np.array([0.0,0.0,0.0]) #np.array([-0.17485121, -0.00969014, -0.02003876]) #np.array([0.0,0.0,0.0])#np.array([-4.79817577e-01 , 3.19565303e-17  ,1.11654641e-17])
all_output_vertices = all_output_vertices - translation



all_output_tensorvertices = torch.Tensor(all_output_vertices)
###############################################################
#### define surface #########

tm_surf = tm_sphere.copy()
tm_surf.vertices = all_output_vertices
#render_trimesh(tm_surf)

########### compute normals, meancurv ##########

batches_diff_quant.compute_curvature()
batches_diff_quant.compute_normals()
normals = batches_diff_quant.all_normals
meancurv = batches_diff_quant.all_H

######### get some functions ##########


#functions, grads, hessians = make_functions_v2()

########## compute LB on functions ##############



functions, grads, hessians = make_functions_v2(fieldstyle = func_type, derivs=True)
evaluated_functions = [f(all_output_tensorvertices) for f in functions]
evaluated_grads = [g(all_output_tensorvertices) for g in grads]
evaluated_hessians = [h(all_output_tensorvertices) for h in hessians]




evaluated_LBs_MC=[]
for i in range(1):
	func, grad_f, hessian_f = evaluated_functions[i], evaluated_grads[i], evaluated_hessians[i]
	evaluated_LBs_MC.append( diffmod.laplace_beltrami_MC(torch.Tensor(normals), torch.Tensor(meancurv), func, grad_f=grad_f, hessian_f=hessian_f) )

	
#	return evaluated_functions, evaluated_grads, evaluated_hessians
	


#evaluated_functions, evaluated_grads, evaluated_hessians = get_function_derivatives(vertices, model, diffmod, batch_size, func_type='fancy_sine')






#analytic (only relevant for sphere)

evaluated_LBs_analy=[]
for i in range(1):
	func, grad_f, hessian_f = evaluated_functions[i], evaluated_grads[i], evaluated_hessians[i]
	
	actual_normals = all_output_tensorvertices
	
	evaluated_LBs_analy.append( diffmod.laplace_beltrami_MC(actual_normals, torch.ones(meancurv.shape[0]), func, grad_f=grad_f, hessian_f=hessian_f) )



LB_MC_errors = []
for i in range(1):

	LB_MC_errors.append(   abs(evaluated_LBs_MC[i] - evaluated_LBs_analy[i])      )
	


############ visualise ###################
#cmap = plt.get_cmap('Spectral')
cmap2 = plt.get_cmap('viridis')






f_colourings=[cmap(linear5(f )) for f in evaluated_functions]
MC_LB_colourings=[cmap(linear7(result)) for result in evaluated_LBs_MC]
#divgrad_LB_colourings=[cmap(linear7(result)) for result in evaluated_LBs_divgrad]
#meshLB_LB_colourings=[cmap(linear7(result)).squeeze() for result in meshLB_results]
analy_LB_colourings=[cmap(linear7(result)).squeeze() for result in evaluated_LBs_analy]


SNS_LB_error_colourings = [ error_cmap(error_scaling(error)).squeeze() for error in LB_MC_errors ]

if show=='show':
	for i in range(1):
	
		for colouring in [ f_colourings[i], MC_LB_colourings[i], analy_LB_colourings[i], SNS_LB_error_colourings[i]]:
			render_trimesh(tm_surf, colouring)

#input()

colouringdict = {'scalarfield':f_colourings[0],
				'mclb':MC_LB_colourings[0],
				#'divgradlb':divgrad_LB_colourings[0],
				#'meshlb':meshLB_LB_colourings[0],
				'analylb':analy_LB_colourings[0],
				'SNS_lb_error': SNS_LB_error_colourings[0]
				}


if not os.path.exists('../data/visualisation/LB/'+surf_name):
	os.makedirs('../data/visualisation/LB/'+surf_name)
	
if not os.path.exists('../data/visualisation/LB_remesh/'+surf_name):
	os.makedirs('../data/visualisation/LB_remesh/'+surf_name)

print('writing ply file')
write_custom_colour_ply_file(tm=tm_surf, colouringdict = colouringdict, filepath='../data/visualisation/LB_remesh/'+surf_name+'/'+surf_name+'_densemesh.ply') 
print('finished writing ply file')






############## originalmesh part ##################


	

#tm_sphere = trimesh.load('../data/analytic/sphere/sphere'+str(level)+'.obj')

#param = torch.load('../data/SNS/'+surf_name+'/param.pth')
#tm_sphere.vertices = param['param']
#tm_sphere.faces = param['faces']
	
#tm_surf_og = tm_sphere.copy()
#tm_surf_og.vertices = param['points']



tm_surf_og = trimesh.load('../data/'+surf_name+'.obj')



tensorvertices_originalmesh = torch.Tensor(tm_surf_og.vertices)


evaluated_functions = [f(tensorvertices_originalmesh) for f in functions]
evaluated_grads = [g(tensorvertices_originalmesh) for g in grads]
evaluated_hessians = [h(tensorvertices_originalmesh) for h in hessians]


print(evaluated_functions[0].shape)

###### cotan or poly laplace computations ##########


if surf_name[:4]=='dual':
#if True:
	
	#print('DUAL DUAL DUAL')

	import scipy.io
	import scipy.sparse as sp
	
	# Read the sparse matrices in Matrix Market format
	L = scipy.io.mmread("../../polyLaplace/results/" + surf_name + "_L_matrix.txt")  # Converts to Compressed Sparse Column (CSC) format
	M = scipy.io.mmread("../../polyLaplace/results/" + surf_name + "_M_matrix.txt")
	
	# Correct the mass matrix to match the scale of the usual triangle mass matrix
	#M *= tm_surf_og.area
	#diagonal = M.diagonal()
	
	#print('trace of M', M.trace())
	#inv_diagonal = 1.0 / diagonal
	#Minv = sp.diags(inv_diagonal)
	
	# Convert sparse matrices to dense format where needed
	#L_poly_matrix = -1 * L.todense() * 200000 / tm_surf_og.area  # Correction
	L_poly_matrix = -1 * L.todense()
	
	
	# Assuming L_cotan() returns cotan-related matrices
	_, Mcotan, Mcotaninv, C = L_cotan(tm_surf_og)
	
	print('C matrix:',C)
	print('trace of M', M.trace())
	L_cotan_matrix = (Mcotaninv @ C).todense()
	
	
	print('traces', L_poly_matrix.trace(), L_cotan_matrix.trace())
	print('traces ratio:',  L_cotan_matrix.trace()/ L_poly_matrix.trace() )
	
	print('surface area', tm_surf_og.area)
	print('num_vertices', tm_surf_og.vertices.shape[0])
	
	
	'''
	from scipy.linalg import eigh
	# For L_poly_matrix
	eigvals_poly = eigh(L_poly_matrix, eigvals_only=True)
	smallest_eigvals_poly = np.sort(abs(eigvals_poly))[:3]
	print('Smallest 3 eigenvalues of L_poly_matrix:', smallest_eigvals_poly)
	
	# For L_cotan_matrix
	eigvals_cotan = eigh(L_cotan_matrix, eigvals_only=True)
	smallest_eigvals_cotan = np.sort(abs(eigvals_cotan))[:3]
	print('Smallest 3 eigenvalues of L_cotan_matrix:', smallest_eigvals_cotan)
	'''
	
	
	print('matrices', L_poly_matrix, L_cotan_matrix)
	
	

	#print(L_matrix)
	print('read polygonal LB matrix, which has shape', L.shape)
	meshLB_results = [ (L_poly_matrix @ f.detach().numpy()).squeeze() for f in evaluated_functions ]
	
	#plt.plot(meshLB_results[0])
	#plt.show()
	
else:

	L,M,Minv,C = L_cotan(tm_surf_og)
	print('trace of M', M.trace())
	sparse_L_cotan_matrix = Minv @ C
	L_cotan_matrix = sparse_L_cotan_matrix.todense()
	
	meshLB_results = [ (L_cotan_matrix @ f.detach().numpy()).squeeze() for f in evaluated_functions ]
	





evaluated_LBs_analy=[]
for i in range(1):
	func, grad_f, hessian_f = evaluated_functions[i], evaluated_grads[i], evaluated_hessians[i]
	
	actual_normals = torch.Tensor(tm_surf_og.vertices)
	
	evaluated_LBs_analy.append( diffmod.laplace_beltrami_MC(actual_normals, torch.ones(actual_normals.shape[0]), func, grad_f=grad_f, hessian_f=hessian_f) )
	
	



##################### SCALE CORRECTION TO THE MESHLB FOR POLY CASE #######################
if surf_name[:4]=='dual':
	print('POLY')
	for i in range(1):
		
		print(evaluated_LBs_analy[i].shape,meshLB_results[i].shape )
		
		
		
		ratios = ( evaluated_LBs_analy[i].squeeze() / torch.tensor(meshLB_results[i]).squeeze() )
		
		#print(ratios)
		
		scale_correction = ratios.mean().item()
		#print(scale_correction)
		
		
		#scale_correction = (  abs(evaluated_LBs_analy[i].squeeze()) / abs(torch.tensor(meshLB_results[i]).squeeze()) ).mean().item()
		
		
		#scale_correction = ( evaluated_LBs_analy[i].squeeze()[0] / torch.tensor(meshLB_results[i]).squeeze()[0] ).item()
		
		
		#scale_correction = 850000/tm_surf_og.area
		
		
		
		print('scale correction', scale_correction)
		meshLB_results[i] = meshLB_results[i] * scale_correction
		
		print(meshLB_results[i][:10])
		print(evaluated_LBs_analy[i][:10])
		#plt.plot(np.array(meshLB_results[i][:10]), 'r-')
		#plt.plot(evaluated_LBs_analy[i][:10], 'k.')
		#plt.show()

##########################################################################################





meshlb_errors = []
for i in range(1):
	meshlb_errors.append(   abs(torch.Tensor(meshLB_results[i]) - evaluated_LBs_analy[i])      )




############ visualise ###################
#cmap = plt.get_cmap('Spectral')
cmap2 = plt.get_cmap('viridis')

f_colourings=[cmap(linear5(f )) for f in evaluated_functions]
#MC_LB_colourings=[cmap(linear7(result)) for result in evaluated_LBs_MC]
#divgrad_LB_colourings=[cmap(linear7(result)) for result in evaluated_LBs_divgrad]
meshLB_LB_colourings=[cmap(linear7(result)).squeeze() for result in meshLB_results]
analy_LB_colourings=[cmap(linear7(result)).squeeze() for result in evaluated_LBs_analy]


mesh_LB_error_colourings = [ error_cmap(error_scaling(error)).squeeze() for error in meshlb_errors ]


print('showing field then meshLB then analyLB then analyLB then meshLB error' )
if show=='show':
	for i in range(1):
		for colouring in [ f_colourings[i], meshLB_LB_colourings[i], analy_LB_colourings[i], mesh_LB_error_colourings[i] ]:
			#print (colouring)
			render_trimesh(tm_surf_og, np.array([0.0,0.0,0.0]))
			render_trimesh(tm_surf_og, colouring)



colouringdict = {#'scalarfield':f_colourings[0],
				#'mclb':MC_LB_colourings[0],
				#'divgradlb':divgrad_LB_colourings[0],
				'meshlb':meshLB_LB_colourings[0],
				#'analylb':analy_LB_colourings[0],
				'meshlb_error':mesh_LB_error_colourings[0]
				}



print('writing ply file')
write_custom_colour_ply_file(tm=tm_surf_og, colouringdict = colouringdict, filepath='../data/visualisation/LB_remesh/'+surf_name+'/'+surf_name+'_originalmesh.ply') 
print('finished writing ply file')












