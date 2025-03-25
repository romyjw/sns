import hakowan as hkw
import lagrange
import math


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib import cm


import numpy as np
import sys
import os


import trimesh
import torch

from utils.custom_ply_writing import *

from visuals.helpers.visualisation_functions import *
from visuals.helpers import rd_helper

#SNSname = sys.argv[-1]


#filenames = ['SMALLTREE', 'SMALLTREE4', 'SMALLTREE5', 'SMALLTREE6']
filenames = ['SMALLTREE']


filepaths = [ ( '../data/plots/'+ filename +'.obj', '../data/plots/'+filename+'_errors.pth' ) for filename in filenames ]

cmap = plt.get_cmap('gist_yarg')



cmap = plt.get_cmap('Reds')

def linear1(x):
	return 50000*x
	
def linear1_inverse(x):
	return x / 50000
	
def linear3(x):
	return 200*x
	
def linear3_inverse(x):
	return x / 200
	

def linear4(x):
	return 0.5*x
	
def linear4_inverse(x):
	return x / 0.5
	
	

def linear2(x):
	return x / 30
	

def linear2_inverse(x):
	return 30 * x

def identity(x):
	return x

geom_map = linear3
geom_map_inverse = linear3_inverse


H_map = linear4
H_map_inverse = linear4_inverse

K_map = linear4
K_map_inverse = linear4_inverse

mincurvdir_map = linear2
mincurvdir_map_inverse = linear2_inverse

normals_map = linear2
normals_map_inverse = linear2_inverse



# Create a range of values to map to colorbars (0 to 1 normalized input)
values = np.linspace(0, 1, 100)

# Map the values using geom_map and H_map
geom_mapped_values = geom_map_inverse(values)
H_mapped_values = H_map_inverse(values)
K_mapped_values = H_map_inverse(values)
mincurvdir_mapped_values = mincurvdir_map_inverse(values)
normals_mapped_values = normals_map_inverse(values)


# Set up the color normalization based on the mapped value ranges
geom_norm = matplotlib.colors.Normalize(vmin=np.min(geom_mapped_values), vmax=np.max(geom_mapped_values))
H_norm = matplotlib.colors.Normalize(vmin=np.min(H_mapped_values), vmax=np.max(H_mapped_values))
K_norm = matplotlib.colors.Normalize(vmin=np.min(K_mapped_values), vmax=np.max(K_mapped_values))
mincurvdir_norm = matplotlib.colors.Normalize(vmin=np.min(mincurvdir_mapped_values), vmax=np.max(mincurvdir_mapped_values))
normals_norm = matplotlib.colors.Normalize(vmin=np.min(normals_mapped_values), vmax=np.max(normals_mapped_values))


# Create a figure with two subplots (for two colorbars)
fig, ax = plt.subplots(1, 5, figsize=(10, 3))

# Create the colorbars using the 'Reds' colormap
cbar1 = cm.ScalarMappable(norm=geom_norm, cmap=cmap)
cbar2 = cm.ScalarMappable(norm=H_norm, cmap=cmap)
cbar3 = cm.ScalarMappable(norm=mincurvdir_norm, cmap=cmap)
cbar4 = cm.ScalarMappable(norm=K_norm, cmap=cmap)
cbar5 = cm.ScalarMappable(norm=normals_norm, cmap=cmap)


# Add colorbars to the plots
fig.colorbar(cbar1, ax=ax[0], orientation='vertical', label='geom_error')
fig.colorbar(cbar2, ax=ax[1], orientation='vertical', label='H_error')
fig.colorbar(cbar3, ax=ax[2], orientation='vertical', label='mincurvdir_error')
fig.colorbar(cbar4, ax=ax[3], orientation='vertical', label='K_error')
fig.colorbar(cbar5, ax=ax[4], orientation='vertical', label='normals_error')


# Add titles
ax[0].set_title('geom_error Colorbar')
ax[1].set_title('H_error Colorbar')
ax[2].set_title('mincurvdir_error Colorbar')
ax[3].set_title('K_error Colorbar')
ax[4].set_title('normals_error Colorbar')


# Display the colorbars
plt.tight_layout()
plt.show()













	
for meshpath, errorpath in filepaths:
	
	errors = torch.load(errorpath)
	print(errors.keys())
	geom_error = errors['geom']['SNS']
	H_error = errors['H']['SNS']
	K_error = errors['K']['SNS']
	normals_error = errors['normals']['SNS']
	mincurvdir_error = errors['mincurvdir']['SNS']
	
	
	
	
	colouringdict = {'geom_error_colour':cmap(geom_map(geom_error)),
						'H_error_colour':cmap(H_map(H_error)),
						'mincurvdir_error_colour':cmap(mincurvdir_map(mincurvdir_error)),
						'K_error_colour':cmap(K_map(K_error)),
						'normals_error_colour':cmap(normals_map(normals_error))}
	
	ply_meshpath = meshpath[:-4]+'.ply'
	
	tm = trimesh.load(meshpath)
	write_custom_colour_ply_file(tm=tm, colouringdict=colouringdict, filepath=ply_meshpath)
	
	for colouringname, colour in colouringdict.items():
		print(colouringname, meshpath)
		
		#render_trimesh(tm, colour)
		
		
	
for meshpath, errorpath in filepaths:

	ply_meshpath = meshpath[:-4]+'.ply'
	
	for colouringname in ['mincurvdir_error_colour', 'H_error_colour', 'normals_error_colour', 'K_error_colour', 'geom_error_colour']:
    
		base =  hkw.layer(ply_meshpath).material(
		"Principled",
		color=hkw.texture.ScalarField(
		   data=colouringname, colormap='identity'
		),
		roughness=0.6,
		).rotate((1,0,0), 3.14 * (-3/8))
		
		
		# Step 2: Adjust camera position.
		config = hkw.config()
		
		config.sensor.location = [0, 1, 5]
		
		#config.film.width = 3840
		#config.film.height = 2160
		
		#config.albedo_only=True
		
		# Step 3: Render the image.
		img_filename = meshpath[:-4]+colouringname+'.png'
		hkw.render(base, config, filename=img_filename)
		
