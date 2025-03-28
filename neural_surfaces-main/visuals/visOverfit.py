import sys
from runners import MainRunner
from mains.experiment_configurator import ExperimentConfigurator
import torch

import numpy as np
import trimesh
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

from visuals.helpers.colourmappings import *


######################################################################################
###################################### Settings ######################################


surf_name = sys.argv[-2]
write_ply=False #Make True if you need custom .ply files, storing the various colourings
reject_points_for_PCD=False #Make True to do rejection sampling to make the crossfields approximately uniformly spaced
target_number_samples=10000 #Approximate number of crossfield samples, if doing rejection sampling


level = sys.argv[-1]### icosphere mesh resolution/ level

#which quantities to visualise
#available quantities are: distortion, directions, maxabscurv, meancurv, gausscurv, normals, default, beltrami_on_X, beltrami_H
#beltrami_on_X computes Delta X (=Hn)
#beltrami_H computes H, via Delta X = Hn
#use ['default'] if you just want to see the shape without special colouring.
vis_settings = ['normals', 'gausscurv', 'meancurv', 'directions']


# reduce batch size if you experience memory issues
batch_size = 20000
subdiv_steps = 0 #if >0, the sphere mesh is subdivided further before pushing throught the SNS


### Principal Curvature Direction (PCD) settings
arrow_length = 0.005
offset_factor = 0.01 #how much offset from the surface, in the normal direction
ratio = 10.0 #controls how 'pointy' the arrows are



cmap_name = 'seismic'
curv_cmap = plt.get_cmap(cmap_name)
dist_cmap = plt.get_cmap(cmap_name)

####### select mappings for more control over colourmaps. Definitions are in visuals/helpers/colourmappings.py
Hmap = linear9
Kmap = linear5
#Hmap = linear
#Kmap = quadratic
#Hmap = mapping12#linear #(suggestions: spike:linear, armadillo:mapping12, igea:linear)
#Kmap = mapping13#quadratic #(suggestions: spike:quadratic, armadillo:mapping13, igea:quadratic)
distmap = logmap

def normals_cmap(normals):
	ans = 0.5*normals + 0.5
	return ans

##################################################################################
##################################################################################

def replace_nans(arr, value=0.0):
	arr[np.isnan(arr)] = value
	return arr


diffmod = DifferentialModule()

## use this in most cases
modules_creator = ExperimentConfigurator()
runner = MainRunner('experiment_configs/overfit/ARMADILLO21622.json', modules_creator)## runner needs to be initialised with some (any) json expmt file.
model = runner.get_model()


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
	
	
	
	
	
model.load_state_dict(weights)
model.eval()

tm_sphere = trimesh.load('../data/analytic/sphere/sphere'+str(level)+'.obj')



vertices = tm_sphere.vertices
faces = tm_sphere.faces
for i in range(subdiv_steps):
	vertices, faces = refine(vertices, faces, spherical=True) #one subdivision step

from differential import batches_diff_quant
batches_diff_quant = batches_diff_quant.batches_diff_quant(vertices, model, diffmod, batch_size)

batches_diff_quant.compute_SNS()
if 'meancurv' in vis_settings or 'gausscurv' in vis_settings:
	batches_diff_quant.compute_curvature()
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


tm_surf = tm_sphere.copy()
tm_surf.vertices = all_output_vertices



#### make a list of colourings to display in pyrender ###

		
colouringdict = {
	'colour_normals' : normals_cmap(all_normals),
	'colour_meancurv' : curv_cmap(Hmap(all_H)),
	'colour_gausscurv' : curv_cmap(Kmap(all_K)),
	'colour_beltrami_H' : curv_cmap(Hmap(all_beltrami_H)),
	'colour_area_distortion' : dist_cmap(distmap(all_area_distortions )),
	'colour_beltrami_on_X' : scaled_normals_cmap2(all_beltrami_on_X)
	}

######################### export as obj #######
tm_surf.export('../data/visualisation/'+surf_name+'/icosphere_'+surf_name+str(level)+'.obj')


############# visualise in pyrender ###########
render_trimeshes([tm_surf], [np.array([0.0,0.5,0.5]), np.array([0.8,0.0,0.0])] )
for key, colouring in colouringdict.items():
	if key[7:] in vis_settings:
		print('Displaying the '+key+' colouring.')
		render_trimesh(tm_surf, colouring)
		render_trimesh(tm_sphere, colouring)



#################### Save colourings in custom ply files (useful for rendering, especially in hakowan https://hakowan.github.io/hakowan/) ###################

if not os.path.exists('../data/visualisation/'+surf_name):
	os.makedirs('../data/visualisation/'+surf_name)
	
if write_ply:    
	write_custom_colour_ply_file(tm=tm_surf, colouringdict = colouringdict, filepath='../data/visualisation/'+surf_name+'/'+surf_name+'.ply') 





################## Code for generating crossfield obj files ######################
if 'directions' in vis_settings:
	# Create mesh objects for the vectors (lines or arrows)
	dir1=all_directions[0]
	dir2=all_directions[1]
	
	
	from .helpers.crossfield import *
	
	
	if reject_points_for_PCD==True:
		area_density = all_area_distortions
		sample_points, area_density, keeping_iis = rejection_sampling(torch.Tensor(tm_surf.vertices), torch.Tensor(area_density), target_number_samples) 
		
		sample_normals = all_normals[keeping_iis,:]
		sample_dir1 = dir1[keeping_iis,:]
		sample_dir2 = dir2[keeping_iis,:]
		sample_points = sample_points.detach().numpy()
	else:
		sample_normals = all_normals
		sample_dir1 = dir1
		sample_dir2 = dir2
		sample_points = tm_surf.vertices
	
	writequadsF(sample_points, sample_dir1, sample_dir2, offset_factor, sample_normals, arrow_length, ratio, overlap=True, filename=surf_name+'/crossfield.obj')
	writequadsF(sample_points, sample_dir1, sample_dir2, offset_factor, sample_normals, arrow_length, ratio, overlap=True, filename=surf_name+'/crossfield_min_dir.obj', which=['B','D'])
	writequadsF(sample_points, sample_dir1, sample_dir2, offset_factor, sample_normals, arrow_length, ratio, overlap=True, filename=surf_name+'/crossfield_max_dir.obj',which=['A','C'])
	print('wrote crossfields')







############################# draw the colourbars for curvatures #####################

# Gauss Curvature Colourbar
values = np.linspace(-700, 700, 15)

transformed_values = [Kmap(val) for val in values]
fig, ax = plt.subplots()
cbar = ColorbarBase(ax, cmap=cmap_name, norm=Normalize(vmin=0, vmax=1))
cbar.set_ticks(transformed_values)
cbar.set_ticklabels([f'{t:.2f}' for t in values])
plt.title('Gauss Curvature Colourbar')
plt.show()

# Mean Curvature Colourbar
values = np.linspace(-32, 32, 17)
transformed_values = [Hmap(val) for val in values]
fig, ax = plt.subplots()
cbar = ColorbarBase(ax, cmap=cmap_name, norm=Normalize(vmin=0, vmax=1))
cbar.set_ticks(transformed_values)
cbar.set_ticklabels([f'{t:.2f}' for t in values]) 
plt.title('Mean Curvature Colourbar')
plt.show()

