from runners import MainRunner
from mains.experiment_configurator import ExperimentConfigurator
import torch
import numpy as np
import trimesh#to load mesh
import igl
import pyrender
from .helpers.visualisation_functions import *
from differential import *
from .helpers import rd_helper
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from scipy import sparse
import sys
from matplotlib import pyplot as plt

#from laplace.mesh_LB import L_cotan, L_uniform, laplace_beltrami_cotan_MC

from utils.custom_ply_writing import *

def visualise_pointcloud(vertices, colors):

	ptcloud_mesh = pyrender.Mesh.from_points(vertices, colors=colors)
	
	show_mesh_gui([ptcloud_mesh])#Display meshes.
	


modules_creator = ExperimentConfigurator()
runner = MainRunner('experiment_configs/eigenfunc/test.json', modules_creator)
model = runner.get_model()

diffmod = DifferentialModule()



######################## options #################################
show_cotan=True
show_samples=False
plotting=False
diff_view=True

write_ply=False
cmap = plt.get_cmap('afmhot')

#surf_name = 'treefrog9919'#'tr_reg_qes_090'#'scan_018'
surf_name = 'SPIKE'#'tr_reg_qes_090'#'scan_018'
eigenfunc_num = 100
level=6
jumpsize=1




sphere_mesh_path = '../data/analytic/sphere/sphere'+str(level)+'.obj'
surface_mesh_path = '../data/icosphere_'+surf_name+str(level)+'.obj'#'../data/icosphere_bottle6.obj'
sphere_tm = trimesh.load(sphere_mesh_path)
sphere_vertices = sphere_tm.vertices

sphere=False
num_eigenfuncs=250

mlp_function_scaling=1

discrete_function_scaling=50000
batch_size = 2000



##################################################################################
colouringdict={}

for j in range(0,eigenfunc_num,jumpsize):
	weights_name = 'model_weights_'+str(j)+'.pth'#'final1steigfunc.pth'#'16aprilmax2.pth'

	#weights = torch.load('../data/heat/treefrog20may/'+weights_name, map_location=torch.device('cpu'))
	#weights = torch.load('../data/heat/camera19may/'+weights_name, map_location=torch.device('cpu'))
	#weights = torch.load('../data/heat/camera3jun25/'+weights_name, map_location=torch.device('cpu'))
	#weights = torch.load('../data/heat/'+surf_name+'/max30sep100000/'+weights_name, map_location=torch.device('cpu'))
	
	
	weights = torch.load('../data/heat/spikey999/'+weights_name, map_location=torch.device('cpu'))



	model.load_state_dict(weights)
	model.eval()
	
	all_output_f_values = np.zeros(sphere_vertices.shape[0])
	
	for i in range(sphere_vertices.shape[0]//batch_size + 1):
	
	
		sphere_tensorvertices = torch.Tensor(sphere_vertices[batch_size*i : min(batch_size*(i+1), sphere_vertices.shape[0]), :])
		sphere_tensorvertices.requires_grad = True

		output_f_values = model.forward(sphere_tensorvertices).squeeze().mean(-1)
			
		all_output_f_values[batch_size*i : min(batch_size*(i+1), sphere_vertices.shape[0])] = output_f_values.detach().numpy().copy()
	
	
	surf_tm = trimesh.load(surface_mesh_path)
	
	if diff_view==True:
		surf_tm.vertices = -1*surf_tm.vertices[:,(2,1,0)]
	
	if sphere==True:
		surf_tm.vertices = ((surf_tm.vertices).T * abs(all_output_f_values)).T
	
	#colouring = cmap(mlp_function_scaling * all_output_f_values + 0.5)
	
	colouring = cmap(all_output_f_values + 0.5)
		
	surf_tm.visual.vertex_colors=colouring
	
	mesh_rd1 = pyrender.Mesh.from_trimesh(surf_tm)
	show_mesh_gui([mesh_rd1])
	
	#plt.plot(all_output_f_values)
	#plt.show()
	
	colouringdict['frame_'+str(j)] = colouring.copy()
	
	#plt.plot(all_output_f_values)
	#plt.show()

############ write special ply colourings ########


print(colouringdict)


if write_ply:    
	write_custom_colour_ply_file(tm=surf_tm, colouringdict = colouringdict, filepath='../data/visualisation/heat/SPIKE/heatflow.ply') 








