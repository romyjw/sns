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
write_ply=False
reject_points_for_PCD=False
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




compute_analytic_curvatures = True

data_type = 'mesh'
level = sys.argv[-1]### sphere level, if using subdivided sphere
#vis_settings = ['normals', 'meancurv', 'directions']#['beltrami_on_X', 'beltrami_H']#['normals', 'meancurv', 'gausscurv']#distortion, directions, maxabscurv, meancurv, gausscurv, normals, default
vis_settings = ['normals', 'gausscurv', 'meancurv', 'directions']



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
#arrow_length = 0.0035
#offset_factor = 0.01
#ratio = 10.0
	
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




############# visualise in pyrender ###########
render_trimeshes([tm_surf], [np.array([0.0,0.5,0.5]), np.array([0.8,0.0,0.0])] )
for key, colouring in colouringdict.items():
	if key[7:] in vis_settings:
		print('Displaying the '+key+' colouring.')
		render_trimesh(tm_surf, colouring)
		render_trimesh(tm_sphere, colouring)



#write_custom_colour_ply_file(tm=tm_surf, colouringdict = {}, filepath='../data/visualisation/treefrog9919/shape599.ply')


#write_custom_colour_ply_file(tm=tm_surf, colouringdict = {'scalar':colouringdict['colour_area_distortion']}, filepath='../data/visualisation/'+surf_name+'/'+surf_name+'scalar.ply')

#write_custom_colour_ply_file(tm=tm_sphere, colouringdict = {'scalar':colouringdict['colour_area_distortion']}, filepath='../data/visualisation/'+surf_name+'/'+surf_name+'scalar_sphere.ply')

if not os.path.exists('../data/visualisation/'+surf_name):
	os.makedirs('../data/visualisation/'+surf_name)
	
if write_ply:    
	write_custom_colour_ply_file(tm=tm_surf, colouringdict = colouringdict, filepath='../data/visualisation/'+surf_name+'/'+surf_name+'.ply') 


######################### export as obj #######
tm_surf.export('../data/icosphere_'+surf_name+str(level)+'.obj')

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


if compute_analytic_curvatures:
	
	with open('../data/analytic/'+surf_name+'/analytic_result.txt') as rf:
		matlab_equations = rf.read()
	
	numpy_equations_list = matlab_to_numpy(matlab_equations)
		
	tm_analytic = trimesh.load('../data/analytic/'+surf_name+'/mesh'+str(level)+'.obj')
	vertices = tm_analytic.vertices
	sph_vertices = (vertices.T/np.linalg.norm(vertices,axis=1).T).T
	
	# spherical polar coordinates
	theta = np.arccos(sph_vertices[:,2].clip(-1,1))
	phi = np.arctan2(sph_vertices[:,1], sph_vertices[:,0])
	scale = 0.7858580264979684 #0.7858563266857805
	#tm_analytic.vertices *= scale
	for statement in numpy_equations_list:
		exec(statement)# compute/define H,K,n1,n2,n3, dir11, dir12, dir13, dir21, dir22, dir23
	
	##### correct sign #########
	H *=-1
	analytic_H = H
	analytic_K = K
	##### construct normals array ####
	analytic_normals = replace_nans( np.stack([n1,n2,n3]).transpose() )
	analytic_dir1 = replace_nans ( np.stack([dir11,dir12,dir13]).transpose() )
	analytic_dir2 = replace_nans ( np.stack([dir21,dir22,dir23]).transpose() )
	
	#### correct the order of min and max curvature directions ####
	analytic_dir1, analytic_dir2 = (analytic_dir2, analytic_dir1)
	
	#### normalize the directions ####
	analytic_dir1 = replace_nans( analytic_dir1.transpose() / np.linalg.norm(analytic_dir1, axis=1) ).transpose()
	analytic_dir2 = replace_nans( analytic_dir2.transpose() / np.linalg.norm(analytic_dir2, axis=1) ).transpose()
	
	## fix signs
	frame = (np.stack([analytic_dir1, analytic_dir2, analytic_normals])).transpose((1,0,2))
	print(frame.shape)
	signs = np.linalg.det(frame)
	analytic_dir1 = (analytic_dir1.T*signs.T).T
	

	mapped_meancurv = replace_nans( Hmap(analytic_H) )
	mapped_gausscurv = replace_nans( Kmap(analytic_K) ) 

	Hcolours = curv_cmap(mapped_meancurv)
	Kcolours = curv_cmap(mapped_gausscurv)
	normalcolours = normals_cmap(analytic_normals) 	
	from .helpers.crossfield import *
	
	writequadsF(vertices, analytic_dir1, analytic_dir2, offset_factor, analytic_normals, arrow_length, ratio, overlap=True, filename='analyticSPIKE/crossfield_both_dir.obj')
	writequadsF(vertices, analytic_dir1, analytic_dir2, offset_factor, analytic_normals, arrow_length, ratio, overlap=True, filename='analyticSPIKE/crossfield_min_dir.obj', which=['A','C'])
	writequadsF(vertices, analytic_dir1, analytic_dir2, offset_factor, analytic_normals, arrow_length, ratio, overlap=True, filename='analyticSPIKE/crossfield_max_dir.obj',which=['B','D'])
	
	
	print(normalcolours.shape)
	
	########### show with pyrender ################
	
	colouringdict = {
	'colour_normals' : normalcolours,
	'colour_meancurv' : Hcolours,
	'colour_gausscurv' : Kcolours,
	'colour_beltrami_H' : np.zeros_like(vertices),
	'colour_area_distortion' : np.zeros_like(vertices),
	'colour_beltrami_on_X' : np.zeros_like(vertices)
	}
	
	
	
	
	'''
	
	for key, colouring in colouringdict.items():
		if key[7:] in vis_settings:
			print('Displaying the analytic '+key+' colouring.')
			render_trimeshes([tm_analytic,tm_surf], [colouring,colouring] )
	'''
	
	
	
			
	############## Compute the errors in the differential quantities ###########
	
	geometry_error = (( tm_analytic.vertices - tm_surf.vertices )**2).sum(1)**(0.5) # looks wrong
	#print('max dimension', np.max(np.abs(tm_surf.vertices)))
	
	normals_angle_error =  np.arccos ( (analytic_normals * all_normals).sum(1) ) * (180 / np.pi)
	normals_angle_error = normals_angle_error[np.isfinite(normals_angle_error)]
	
	min_curv_dir_error = np.arccos ( abs(( analytic_dir1 *  dir2).sum(1)) ) * (180 / np.pi)
	max_curv_dir_error = np.arccos ( abs(( analytic_dir2 *  dir1).sum(1)) ) * (180 / np.pi) 
	H_error = (analytic_H - all_H) 
	K_error = (analytic_K - all_K)
	
	
	##################### Compute error map colours #############################
	
	geometry_error_colours = geometry_error_cmap(geometry_error)
	normals_error_colours = (normals_error_cmap(normals_angle_error))
	min_curv_dir_error_colours = (dir_error_cmap(min_curv_dir_error))
	max_curv_dir_error_colours = (dir_error_cmap(max_curv_dir_error))
	H_error_colours = H_error_cmap(H_error)
	K_error_colours = K_error_cmap(K_error)
	
	##################### plot histograms #######################################
	
	
	############ plot error maps #################################
	
	error_colouringdict = {
		'colour_geometry_error' : geometry_error_colours,
	'colour_normals_error' : normals_error_colours,
	'colour_H_error' : H_error_colours,
	'colour_K_error' : K_error_colours,
	'colour_min_curv_dir_error' : min_curv_dir_error_colours,
	'colour_max_curv_dir_error' : max_curv_dir_error_colours
	}
	
	
	for key, colouring in error_colouringdict.items():
		
		print('Displaying the '+key+' colouring.')
		render_trimesh(tm_analytic, colouring)
	
	######## write a special ply file with the curvature attributes #################
	#write_custom_colour_ply_file(tm=tm_analytic, colouringdict = colouringdict, filepath='../data/visualisation/'+'analytic'+surf_name+'/analytic'+surf_name+'.ply')
	
	#write_curvature_ply_file(tm=tm_analytic, H=H, K=K, Hmap=Hmap, Kmap=Kmap, filepath='../data/visualisation/Analyticwithcurv.ply')
	







############################# colourbar for mapping12 #####################

# Create linearly spaced values for the color bar
values = np.linspace(-700, 700, 15)

# Transform these values using the nonlinear mapping function
transformed_values = [Kmap(val) for val in values]

# Create a figure and axis
fig, ax = plt.subplots()

# Create a ColorbarBase object
cbar = ColorbarBase(ax, cmap=cmap_name, norm=Normalize(vmin=0, vmax=1))

# Set the color bar ticks with the transformed values
cbar.set_ticks(transformed_values)

# Set tick labels using the inverse transformation
cbar.set_ticklabels([f'{t:.2f}' for t in values])  # Example: Square root of the transformed ticks

plt.show()


# Create linearly spaced values for the color bar
values = np.linspace(-32, 32, 17)

# Transform these values using the nonlinear mapping function
transformed_values = [Hmap(val) for val in values]

# Create a figure and axis
fig, ax = plt.subplots()

# Create a ColorbarBase object
cbar = ColorbarBase(ax, cmap=cmap_name, norm=Normalize(vmin=0, vmax=1))

# Set the color bar ticks with the transformed values
cbar.set_ticks(transformed_values)

# Set tick labels using the inverse transformation
cbar.set_ticklabels([f'{t:.2f}' for t in values])  # Example: Square root of the transformed ticks

plt.show()





