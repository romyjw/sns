from runners import MainRunner
from mains.experiment_configurator import ExperimentConfigurator
import torch
import numpy as np
import trimesh#to load mesh
import igl
import pyrender
from .helpers.visualisation_functions import *
from differential import *
from .helpers.rd_helper import *
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from scipy import sparse
import sys

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
show_cotan=False
show_samples=False
plotting=False


write_ply=True
cmap = plt.get_cmap('Spectral')
#cmap = plt.get_cmap('coolwarm')


surf_name = sys.argv[-3]#'tr_reg_qes_090'#'scan_018'
eigenfunc_num = int(sys.argv[-2])
level=int(sys.argv[-1])


if not os.path.exists("../data/visualisation/eigenfunc/"+surf_name):
    os.mkdir("../data/visualisation/eigenfunc/"+surf_name)



sphere_mesh_path = '../data/analytic/sphere/sphere'+str(level)+'.obj'
surface_mesh_path = '../data/analytic/sphere/sphere'+str(level)+'.obj'


#surface_mesh_path = '../data/icosphere_'+surf_name+str(level)+'.obj'#'../data/icosphere_bottle6.obj'

#surf_name = 'scan_018'
#weights_name = 'ortho2.pth'#'final1steigfunc.pth'#'16aprilmax2.pth'
#sphere_mesh_path = '../data/analytic/sphere/sphere6.obj'
#surface_mesh_path = '../data/icosphere_scan_0186.obj'#'../data/icosphere_bottle6.obj'


sphere=True

mlp_function_scaling=1



batch_size = 2000




################ show the samples on the surface #################
if show_samples:
	samples = torch.load('../data/eigenfunc/'+surf_name+'/samples.pth', map_location=torch.device('cpu'))
	points = np.array(samples['points'])
	
	tm = trimesh.Trimesh(vertices=points,faces=[])
	                   
	tm.export('../data/eigenfunc/'+surf_name+'/ptcloud.ply')
	
	visualise_pointcloud(points, np.zeros_like(points))
	print('number of samples: ', points.shape[0])

sphere_tm = trimesh.load(sphere_mesh_path)
sphere_vertices = sphere_tm.vertices

##################################################################################


for j in range(eigenfunc_num):
	colouringdict={}
	weights_name = 'ortho'+str(j+1)+'.pth'#'final1steigfunc.pth'#'16aprilmax2.pth'

	weights = torch.load('../data/eigenfunc/'+surf_name+'/orthoweights/'+weights_name, map_location=torch.device('cpu'))
	model.load_state_dict(weights)
	model.eval()
	
	
	
	
	all_output_f_values = np.zeros(sphere_vertices.shape[0])
	
	for i in range(sphere_vertices.shape[0]//batch_size + 1):
	
	
		sphere_tensorvertices = torch.Tensor(sphere_vertices[batch_size*i : min(batch_size*(i+1), sphere_vertices.shape[0]), :])
		sphere_tensorvertices.requires_grad = True
		try:
			output_f_values = model.forward(sphere_tensorvertices).squeeze()
		except:
			output_f_values = model.forward(sphere_tensorvertices,torch.eye(3)).squeeze()
	
		all_output_f_values[batch_size*i : min(batch_size*(i+1), sphere_vertices.shape[0])] = output_f_values.mean(-1).detach().numpy().copy()
	
	if plotting==True:
		plt.plot(all_output_f_values)
		plt.show()
	
	surf_tm = trimesh.load(surface_mesh_path)
	
	if sphere==True:
		surf_tm.vertices = ((surf_tm.vertices).T * abs(all_output_f_values)).T
	
	colouring = cmap(mlp_function_scaling * all_output_f_values + 0.5)
	#colouring = cmap((surf_tm.vertices[:,1]+0.5)**10)
	surf_tm.visual.vertex_colors=colouring
	
	mesh_rd1 = pyrender.Mesh.from_trimesh(surf_tm)
	show_mesh_gui([mesh_rd1])
	
	colouringdict['eigfunc'] = colouring.copy()

	############ write special ply colourings ########

	if write_ply:    
		write_custom_colour_ply_file(tm=surf_tm, colouringdict = colouringdict, filepath='../data/visualisation/eigenfunc/'+surf_name+'/'+surf_name+str(j+1)+'.ply') 








