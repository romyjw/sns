import sys
from runners import MainRunner
from mains.experiment_configurator import ExperimentConfigurator
import torch

import numpy as np
import trimesh#to load mesh
import igl

import pyrender

from visualisation_functions import *
from torch import autograd as Grad


from scipy.spatial import Delaunay, ConvexHull
from geometric_functions.rotations import compute_rotation_sig, rotate2pole1
from geometric_functions.mobius_triplet import *
#from rotations import 



def visualise_pointcloud(vertices, colors):

	ptcloud_mesh = pyrender.Mesh.from_points(vertices, colors=colors)
	show_mesh_gui([ptcloud_mesh])#Display meshes.
	
	
def delaunay_remesh(vertices, mapped_vertices):


	#### stereographic
	rotation_sig = compute_rotation_sig(vertices) ### Find a rotation that brings vertex 0 to the North Pole
	stereo_vertices = stereographic2(rotate2pole1(vertices[1:, :], rotation_sig)) ### Do stereographic projection to the rotated sphere. Vertex 0 goes to infinity so skip that.
	tri = Delaunay(stereo_vertices)
	tri_simplices = tri.simplices[:, [0,2,1]] + 1
	t = tri_simplices.shape[0]
	
	
	hull_vertices = ConvexHull(stereo_vertices).vertices + 1
	n = hull_vertices.shape[0]
	
	faces = np.zeros((n + t, 3), dtype = 'int32')
	faces[:t, :] = tri_simplices
	
	for i in range(n):
		faces[t + i , :] = [0, hull_vertices[i], hull_vertices[(i+1)%n]]
	
	### take back to R3
	
	igl.write_triangle_mesh('../data/remeshing/meshed_sphere.obj',vertices.detach().numpy(), faces)
	
	
	igl.write_triangle_mesh('../data/remeshing/meshed_surface.obj',mapped_vertices.detach().numpy(), faces)



def random_unit_sphere(num_points):
    # Generate 3D points from a Gaussian distribution
    points = np.random.normal(size=(num_points, 3))
    
    # Normalize each point to lie on the unit sphere
    norms = np.linalg.norm(points, axis=1)  # Calculate the norm (magnitude) of each point
    normalized_points = points / norms[:, np.newaxis]  # Normalize each point
    
    return normalized_points



if __name__ == "__main__":
	index_filename = sys.argv[1]
	vtx_filename = sys.argv[2]
	mapped_vtx_filename = sys.argv[3]
	
	with open(index_filename) as indexfile:
		lines = indexfile.readlines()
		
	iis = [int(line) for line in lines]
	
	
		
	vertices = torch.tensor(trimesh.load(vtx_filename).vertices[iis,:])
	mapped_vertices = torch.tensor ( trimesh.load(mapped_vtx_filename).vertices[iis,:])
	
	
	#vertices = torch.tensor(trimesh.load(vtx_filename).vertices[iis,:])
	#mapped_vertices = torch.tensor(trimesh.load('../data/remeshing/trousers.obj').vertices)
		
		
		
	visualise_pointcloud(mapped_vertices, [0.0,0.0,0.0])
		
		
	delaunay_remesh(vertices, mapped_vertices)




