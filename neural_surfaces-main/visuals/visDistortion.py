import sys
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
from utils.custom_ply_writing import *



cmap_name = 'Spectral'
curv_cmap = plt.get_cmap(cmap_name)
dist_cmap = plt.get_cmap(cmap_name)

geometry_error_cmap = plt.get_cmap('gist_yarg')
normals_error_cmap = plt.get_cmap('gist_yarg')
H_error_cmap = plt.get_cmap('gist_yarg')
K_error_cmap = plt.get_cmap('gist_yarg')
dir_error_cmap = plt.get_cmap('gist_yarg')





discrete_dist_cmap = plt.get_cmap('hot')
angle_dist_cmap = plt.get_cmap('cool')

def discrete_dist_map(dist):
	#return np.exp(-dist*10)
	return 100*dist

def angle_dist_map(dist):
	#return np.exp(-dist*10)
	return dist/10.0


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
distmap = logmap


#Hmap = linear
#Kmap = quadratic



#################################################
#################################################



surf_name = sys.argv[-1]
write_ply=True





import numpy as np

def compute_onering_areas(mesh):
    """
    Computes the one-ring area for all vertices in a trimesh mesh.

    Parameters:
        mesh (trimesh.Trimesh): A trimesh object containing vertices and faces.

    Returns:
        np.ndarray: An array where each entry corresponds to the one-ring area of a vertex.
    """
    # Initialize an array to store the one-ring areas
    vertex_areas = np.zeros(len(mesh.vertices))

    # Iterate over each face and distribute its area to its vertices
    for face in mesh.faces:
        # Get the vertices of the face
        v0, v1, v2 = mesh.vertices[face]

        # Compute the area of the triangle using the cross product
        triangle_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        # Distribute 1/3 of the area to each vertex
        for vertex in face:
            vertex_areas[vertex] += triangle_area / 3.0

    return vertex_areas


def compute_triangle_angles(mesh):
    """
    Computes the angles of each triangle in a trimesh mesh.

    Parameters:
        mesh (trimesh.Trimesh): A trimesh object containing vertices and faces.

    Returns:
        np.ndarray: An array of shape (n_faces, 3), where each row contains the angles (in radians)
                    of the corresponding triangle in the mesh.
    """
    # Initialize an array to store the angles
    triangle_angles = np.zeros((len(mesh.faces), 3))

    # Iterate over each face in the mesh
    for i, face in enumerate(mesh.faces):
        # Get the vertices of the face
        v0, v1, v2 = mesh.vertices[face]

        # Compute the edge vectors
        e0 = v1 - v0  # Edge from v0 to v1
        e1 = v2 - v1  # Edge from v1 to v2
        e2 = v0 - v2  # Edge from v2 to v0

        # Compute the lengths of the edges
        l0 = np.linalg.norm(e0)
        l1 = np.linalg.norm(e1)
        l2 = np.linalg.norm(e2)

        # Use the dot product to compute the angles (cosine rule)
        angle0 = np.arccos(np.clip(np.dot(-e2, e0) / (l2 * l0), -1.0, 1.0))  # Angle at v0
        angle1 = np.arccos(np.clip(np.dot(-e0, e1) / (l0 * l1), -1.0, 1.0))  # Angle at v1
        angle2 = np.arccos(np.clip(np.dot(-e1, e2) / (l1 * l2), -1.0, 1.0))  # Angle at v2

        # Store the angles in the array
        triangle_angles[i,:] = [angle0, angle1, angle2]

    return triangle_angles











######################################### show area distortion ###################################


param_mesh = trimesh.load('../data/'+surf_name+'.obj')
param = torch.load('../data/SNS/'+surf_name+'/param.pth')
mesh = trimesh.Trimesh(vertices=param['points'], faces=param_mesh.faces)

distortion = compute_onering_areas(param_mesh) / compute_onering_areas(mesh)

'''
plt.plot(distortion)
plt.show()


dist_colouring = np.array(discrete_dist_cmap(discrete_dist_map(distortion))) 




render_trimesh(param_mesh, dist_colouring )
render_trimesh(mesh, dist_colouring )

colouringdict = {
	'colour_discrete_distortion' : dist_colouring
	}
	

if not os.path.exists('../data/visualisation/'+surf_name):
	os.makedirs('../data/visualisation/'+surf_name)
	
if write_ply:    
	write_custom_colour_ply_file(tm=mesh, colouringdict = colouringdict, filepath='../data/visualisation/'+surf_name+'/discrete_distortion.ply')
	write_custom_colour_ply_file(tm=param_mesh, colouringdict = colouringdict, filepath='../data/visualisation/'+surf_name+'/discrete_distortion_sphere.ply') 
'''



################ angle distortion #########################



angle_distortion = np.array ( np.abs( compute_triangle_angles(param_mesh) - compute_triangle_angles(mesh) ).sum(-1) / 3.0).clip(0,1)


print(angle_distortion)


plt.plot(angle_distortion)
plt.show()






angle_dist_colouring = np.array(angle_dist_cmap((angle_distortion)))[:,:-1]
#print(dist_colouring.dtype, angle_dist_colouring.dtype)


print(angle_dist_colouring.shape)

render_trimesh(param_mesh, angle_dist_colouring )
render_trimesh(mesh, angle_dist_colouring )

















# Create linearly spaced values for the color bar
values = np.linspace(0.0, 0.01, 10)

# Transform these values using the nonlinear mapping function
transformed_values = [discrete_dist_map(val) for val in values]

# Create a figure and axis
fig, ax = plt.subplots()

# Create a ColorbarBase object
cbar = ColorbarBase(ax, cmap=discrete_dist_cmap, norm=Normalize(vmin=0, vmax=1))

# Set the color bar ticks with the transformed values
cbar.set_ticks(transformed_values)

# Set tick labels using the inverse transformation
cbar.set_ticklabels([f'{t:.2f}' for t in values])  # Example: Square root of the transformed ticks

plt.show()

