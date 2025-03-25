import sys
from runners import MainRunner
from mains.experiment_configurator import ExperimentConfigurator
import torch

import numpy as np
import trimesh#to load mesh
import igl

import pyrender

from .helpers.visualisation_functions import *
from .helpers import rd_helper


from .helpers.subdiv import *
from torch import autograd as Grad


from differential import *



import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

import re
from utils.formula_converter import *
from utils.custom_ply_writing import *
from utils.rejection_sampling import rejection_sampling

from scipy.interpolate import interp1d


cmap_name = 'seismic'
curv_cmap = plt.get_cmap(cmap_name)
dist_cmap = plt.get_cmap(cmap_name)

geometry_error_cmap = plt.get_cmap('gist_yarg')
normals_error_cmap = plt.get_cmap('gist_yarg')
H_error_cmap = plt.get_cmap('coolwarm')
K_error_cmap = plt.get_cmap('coolwarm')
dir_error_cmap = plt.get_cmap('gist_yarg')


def normals_cmap(normals):
	ans = 0.5*normals + 0.5
	#ans[:,0]*=0.0
	#ans[:,2]*=0.0
	return ans

################## colour map stuff #############
#################################################
from visuals.helpers.colourmappings import *


def replace_nans(arr, value=0.0):
	arr[np.isnan(arr)] = value
	return arr
	
	
#Hmap = mapping12#linear #(so far, spike:linear, armadillo:mapping12, igea:linear)
#Kmap = mapping13#quadratic #(so far, spike:quadratic, armadillo:mapping13, igea:quadratic)
distmap = logmap


### for froggo:
Hmap = linear9
Kmap = linear5

#Hmap = linear
#Kmap = quadratic



#################################################
#################################################

diffmod = DifferentialModule()

## use this in most cases
modules_creator = ExperimentConfigurator()
runner = MainRunner('experiment_configs/overfit/ARMADILLO21622.json', modules_creator)## easier than sys.argv[1]
model = runner.get_model()

surf_name = sys.argv[-2]
write_ply=True
reject_points_for_PCD=True
target_number_samples=10000

if surf_name=='TEMP':
	folder_path =  '../checkpoints/AAAtemp'
	pth_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pth') and os.path.isfile(os.path.join(folder_path, f))])
	# Ensure there's at least one .pth file
	if not pth_files:
		raise FileNotFoundError(f"No .pth files found in directory: {folder_path}")

	# Get the first .pth file
	filepath = pth_files[-1]
	weights = torch.load(filepath, map_location=torch.device('cpu'))
	
else:
	weights = torch.load('../data/SNS/'+surf_name+'/weights.pth', map_location=torch.device('cpu'))




compute_analytic_curvatures = False

data_type = 'mesh'
level = sys.argv[-1]### sphere level, if using subdivided sphere
#vis_settings = ['normals', 'meancurv', 'directions']#['beltrami_on_X', 'beltrami_H']#['normals', 'meancurv', 'gausscurv']#distortion, directions, maxabscurv, meancurv, gausscurv, normals, default
vis_settings = ['normals', 'gausscurv', 'meancurv']


batch_size = 20000
subdiv_steps = 0




### PCD settings
arrow_length = 0.005
offset_factor = 0.01
ratio = 10.0

### PCD settings
#arrow_length = 0.00125
#offset_factor = 0.01
#ratio = 10.0

#recommended
arrow_length = 0.0035
offset_factor = 0.01
ratio = 10.0
	
#arrow_length = 0.001
#offset_factor = 0.01
#ratio = 10.0


#arrow_length = 0.06
#offset_factor = 0.00
#ratio = 10.0
	
	

	
##################################################################################

model.load_state_dict(weights)
model.eval()

tm_sphere = trimesh.load('../data/analytic/sphere/sphere'+str(level)+'.obj')


#tm_compare = trimesh.load('../data/'+surf_name+'6.obj')

vertices = tm_sphere.vertices
faces = tm_sphere.faces
for i in range(subdiv_steps):
	vertices, faces = refine(vertices, faces, spherical=True) #one subdivision step

from differential import batches_diff_quant
batches_diff_quant = batches_diff_quant.batches_diff_quant(vertices, model, diffmod, batch_size)

batches_diff_quant.compute_SNS()
if 'meancurv' in vis_settings or 'gausscurv' in vis_settings:
	batches_diff_quant.compute_curvature()
	#pass
if 'normals' in vis_settings:
	batches_diff_quant.compute_normals()
if 'beltrami_H' in vis_settings:
	batches_diff_quant.compute_beltrami_H()
if 'area_distortion' in vis_settings:
	batches_diff_quant.compute_area_distortions()
if 'beltrami_on_X' in vis_settings:
	batches_diff_quant.compute_beltrami_on_X()
if 'directions' in vis_settings:
	if reject_points_for_PCD:
		batches_diff_quant.compute_area_distortions()
	batches_diff_quant.compute_directions()
if vis_settings==[]:
	batches_diff_quant.compute_SNS()
	
############# get the info from the batches computation #######
all_H = batches_diff_quant.all_H
all_K = batches_diff_quant.all_K
all_normals = batches_diff_quant.all_normals
all_directions = batches_diff_quant.all_directions
all_output_vertices = batches_diff_quant.all_output_vertices
all_beltrami_H = batches_diff_quant.all_beltrami_H
all_area_distortions = batches_diff_quant.all_area_distortions
all_beltrami_on_X = batches_diff_quant.all_beltrami_on_X
###############################################################

##### save ####
#torch.save({'H':all_H, 'K':all_K}, '../data/visualisation/treefrog9919/curvature9.pth')
#curv = torch.load('../data/visualisation/treefrog9919/curvature.pth')
#all_H = curv['H']
#all_K = curv['K']



tm_surf = tm_sphere.copy()
tm_surf.vertices = all_output_vertices

print(tm_surf.vertices.max(), tm_surf.vertices.min())
print(tm_surf.vertices.shape)

#### make a list of colourings to display in pyrender ###

		
colouringdict = {
	'colour_normals' : normals_cmap(all_normals),
	'colour_meancurv' : curv_cmap(Hmap(all_H)),
	'colour_gausscurv' : curv_cmap(Kmap(all_K)),
	'colour_beltrami_H' : curv_cmap(Hmap(all_beltrami_H)),
	'colour_area_distortion' : dist_cmap(distmap(all_area_distortions )),
	'colour_beltrami_on_X' : scaled_normals_cmap2(all_beltrami_on_X)
	}






#write_custom_colour_ply_file(tm=tm_surf, colouringdict = {}, filepath='../data/visualisation/treefrog9919/shape599.ply')


#write_custom_colour_ply_file(tm=tm_surf, colouringdict = {'scalar':colouringdict['colour_area_distortion']}, filepath='../data/visualisation/'+surf_name+'/'+surf_name+'scalar.ply')

#write_custom_colour_ply_file(tm=tm_sphere, colouringdict = {'scalar':colouringdict['colour_area_distortion']}, filepath='../data/visualisation/'+surf_name+'/'+surf_name+'scalar_sphere.ply')

if not os.path.exists('../data/visualisation/'+surf_name):
	os.makedirs('../data/visualisation/'+surf_name)
	
if write_ply:    
	write_custom_colour_ply_file(tm=tm_surf, colouringdict = colouringdict, filepath='../data/visualisation/'+surf_name+'/'+surf_name+'.ply') 


######################### export as obj #######
tm_surf.export('../data/icosphere_'+surf_name+str(level)+'.obj')








