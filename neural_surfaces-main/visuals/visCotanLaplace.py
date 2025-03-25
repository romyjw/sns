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

from .helpers import rd_helper

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

import re
from utils.formula_converter import *
from utils.custom_ply_writing import *

from differential import batches_diff_quant
from visuals.helpers.colourmappings import *


from torch.nn import functional as F

surf_name = sys.argv[-1]

#torch.manual_seed(0)
torch.manual_seed(1)

cmap_name = 'Spectral'
curv_cmap = plt.get_cmap(cmap_name)
dist_cmap = plt.get_cmap(cmap_name)


#from laplace.LBv2 import make_functions ### random functions, etc.
from SNS_applications.discrete_laplace.mesh_LB import L_cotan, L_uniform, laplace_beltrami_cotan_MC




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
	
	if fieldstyle=='y02':
		functions = [lambda x : 3*(z.pow(2)) - 1.0]
		
		grads = [lambda x : 6 * z]








#################################################
#################################################



####################### load a sphere mesh #################

tm_surf = trimesh.load('../data/'+surf_name+'.obj')



###NORMALIZE!!!!!
vertices = tm_surf.vertices / np.linalg.norm(tm_surf.vertices, axis=1, keepdims=True)






all_output_tensorvertices = torch.Tensor(vertices)


functions, grads, hessians = make_functions_v2(derivs=True)

evaluated_functions = [f(all_output_tensorvertices) for f in functions]




###### cotan computations ##########



L,M,Minv,C = L_cotan(tm_surf)

Minv = Minv.todense()
C = C.todense()
sparse_L_cotan_matrix = (Minv @ C)
L_cotan_matrix = sparse_L_cotan_matrix

cotan_results = [ (L_cotan_matrix @ f.detach().numpy()).squeeze() for f in evaluated_functions ]





############ visualise ###################
cmap = plt.get_cmap('Spectral')
cmap2 = plt.get_cmap('viridis')

f_colourings=[cmap(linear5(f )) for f in evaluated_functions]

cotan_LB_colourings=[cmap(linear6(result)).squeeze() for result in cotan_results]




################################## igl check ###################

#igl_M = igl.massmatrix(tm_surf.vertices, tm_surf.faces, igl.MASSMATRIX_TYPE_VORONOI) 
#igl_Minv = np.linalg.inv(igl_M)
LB_igl = 2 * Minv @ igl.cotmatrix(tm_surf.vertices, tm_surf.faces)


igl_cotan_results = [ (LB_igl@ f.detach().numpy()).squeeze() for f in evaluated_functions ]

print(igl_cotan_results[0].shape )
plt.plot(igl_cotan_results[0])
plt.show()


iglcotan_LB_colourings=[cmap(linear6(result)).squeeze() for result in igl_cotan_results]



for i in range(1):
	for colouring in [ f_colourings[i], cotan_LB_colourings[i], iglcotan_LB_colourings[i]]:
		#for colouring in [ f_colourings[i], cotan_LB_colourings[i]]:
		tm_surf.visual.vertex_colors = np.array([[0,0,1]]) #reset colours
		render_trimesh(tm_surf, colouring)
		render_trimesh(tm_surf, np.array([0.0,0.0,1.0]))


colouringdict = {'scalarfield':f_colourings[0],
				'cotanlb':cotan_LB_colourings[0]}


if not os.path.exists('../data/visualisation/LB_remesh/'+surf_name):
	os.makedirs('../data/visualisation/LB_remesh/'+surf_name)

print('writing ply file')
write_custom_colour_ply_file(tm=tm_surf, colouringdict = colouringdict, filepath='../data/visualisation/LB_remesh/'+surf_name+'/'+surf_name+'.ply') 
print('finished writing ply file')
