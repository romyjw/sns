import torch #to load scalar fields

import numpy as np
import trimesh#to load mesh
import igl

import pyrender

from .visualisation_functions import *

import matplotlib.pyplot as plt


from scipy.spatial import KDTree as scipy_kdtree


show_on_subdiv = True #show colouring on subdivided surface vertices, or on original mesh
curvature_type = 'meancurv' #gausscurv or meancurv




####################################### load the sphere embedding ####################################
tmSph = trimesh.load('../data/armadillo_final_embedding.obj')
#tmSph = trimesh.load('../data/subdiv_armadillo_sphere.obj')

sphere_vertices = tmSph.vertices
faces = tmSph.faces

####################################### load the surface mesh ########################################
tm = trimesh.load('../data/ARMADILLO21622_nA.obj')
#tm = trimesh.load('../data/subdiv_armadillo.obj')
vertices = tm.vertices

####################### load the subdivided surface mesh ############################################
#tm = trimesh.load('../data/ARMADILLO21622_nA_subdiv.obj')
dense_tm = trimesh.load('../data/subdiv_armadillo.obj')
dense_vertices = dense_tm.vertices


######################################## load the scalar field #######################################
if curvature_type=='gausscurv':
	scalar_field = torch.load('/Users/romywilliamson/Desktop/gausscurv_subdiv.pt')
elif curvature_type == 'meancurv':
	scalar_field = torch.load('/Users/romywilliamson/Desktop/meancurv_subdiv.pt')

print(scalar_field.shape)

################### choose colourmap ###########################

if curvature_type=='meancurv':
	cmap = plt.get_cmap('coolwarm')
elif curvature_type=='gausscurv':
	cmap = plt.get_cmap('coolwarm')
	
	
#cmap = plt.get_cmap('PiYG')

#cmap = plt.get_cmap('hsv')
#cmap = plt.get_cmap('gist_rainbow')
#cmap = plt.get_cmap('Spectral')
#cmap = plt.get_cmap('PuOr')
#cmap = plt.get_cmap('RdYlGn')

def interpolate(dense_scalar_field, dense_vertices, sparse_vertices,k=500):

	sparse_scalar_field = np.zeros(sparse_vertices.shape[0])
	kdtree = scipy_kdtree(dense_vertices)
	
	for i in range(sparse_vertices.shape[0]):
		distances, nbr_indices = kdtree.query(sparse_vertices[i,:], k)###
		#print(nbr_indices.shape)
		
		sparse_scalar_field[i] = np.median(dense_scalar_field[nbr_indices])
	
	return sparse_scalar_field
		
def make_test_scalar_field(vertices):
	scalar_field = np.zeros(vertices.shape[0])
	for i in range(vertices.shape[0]):
		if vertices[i,1]>=0:
			scalar_field[i] = 0.5
		else:
			scalar_field[i] = 1.5
	
	return scalar_field		



if show_on_subdiv==False:
	scalar_field = interpolate(scalar_field, dense_vertices, vertices) #convert from big scalar field to interpolated scalar field
else:
	if curvature_type=='gausscurv':
		pass
		#scalar_field = interpolate(scalar_field, dense_vertices, dense_vertices,k=20) # only interpolate for gausscurv, not meancurv
	vertices = dense_vertices



######################## mapping functions to try ####################################

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-1*x) )
	

def mapping1(field):
	const = 0.01*np.max(abs(field))
	return np.array([ field[i]/ const for i in range(vertices.shape[0])])
	
def mapping2(field):
	const = 0.00001*np.max(abs(field))
	return np.array([ field[i]/ const for i in range(vertices.shape[0])])
	
def mapping3(field):
	return np.array([ np.arctan(field[i]) for i in range(vertices.shape[0])])



def mapping4(field):
	return np.array([ sigmoid(0.15*field[i]) for i in range(vertices.shape[0])])
	
def mapping5(field):
	return np.array([ (np.arctan(0.01*field[i])**3 ) for i in range(vertices.shape[0])])

def mapping6(field):
	return np.array([1.0* sigmoid(0.06*field[i]) for i in range(vertices.shape[0])])

def mapping7(field):
	return np.array([ 0.5 + 0.5 * np.tanh(0.01*field[i]) for i in range(vertices.shape[0])])

def mapping8(field):
	return np.array([ 0.5 + 0.5 * np.tanh(0.015*field[i]) for i in range(vertices.shape[0])])
	



def mapping9(field):
	clipped_field = np.clip(field, -100,100)
	return np.array([ 0.5 + 0.5 * np.tanh(clipped_field[i]/100) for i in range(vertices.shape[0])])

def mapping10(field):
	clipped_field = np.clip(field, -75,75)
	return np.array([ 0.5 + 0.5 * 1.25*np.tanh(clipped_field[i]/50) for i in range(vertices.shape[0])])
	
#def mapping11(field):
#	clipped_field = np.clip(field, -75,75)
#	return np.array([ 0.5 + 0.5 * 1.5*np.tanh(field[i]/200) for i in range(vertices.shape[0])])

def mapping12(field):
	return np.array([ sigmoid(0.1*field[i]) for i in range(vertices.shape[0])])
	
def mapping12B(field):
	return np.array([ sigmoid(0.01*field[i]) for i in range(vertices.shape[0])])
	

def mapping12M(field):
	return np.array([ sigmoid(1.5*np.sign(field[i])*abs(field[i])**0.1) for i in range(vertices.shape[0])])
	
def mapping8M(field):#M for mean curvature
	return np.array([ 0.5 + 0.5 * np.tanh(5*np.tanh(field[i])) for i in range(vertices.shape[0])])
	
def mapping9M(field):#M for mean curvature
	mapping = np.array([ 0.5 + 0.5 * np.tanh(5*np.tanh(12*field[i])) for i in range(vertices.shape[0])])
	return mapping








############################# define stupid scalar field
#scalar_field = make_test_scalar_field(vertices)
#with open('/Users/romywilliamson/Desktop/test_field.txt','w') as file:
#	for i in range(vertices.shape[0]):
#		file.write(str(scalar_field[i]))
#		if i!=vertices.shape[0] - 1:
#			file.write('\n')

#colours = cmap(scalar_field) ############ basic one for the simple scalar field

if curvature_type == 'gausscurv':
	colours = cmap(mapping12(scalar_field)) ############ change the choice of function here
	
elif curvature_type == 'meancurv':
	scalar_field*=-1

	plt.plot(scalar_field)
	plt.show()
	colours = cmap(mapping12M(scalar_field)) 
	print(colours.shape)
	
#colours = cmap(0.5 + 0.5*np.sin(  30* (vertices*np.array([1,1,0]) ).sum(-1)  )) ###stripy armadillo test
#colours = cmap(0.5 + 0.5*np.sin(   10*	(vertices*np.array([0,1,0]) ).sum(-1)  )) ###stripy armadillo test



#################################### visualise ###################################
if show_on_subdiv==False:
	tm.visual.vertex_colors= colours
	mesh_rd1 = pyrender.Mesh.from_trimesh(tm)
	show_mesh_gui([mesh_rd1])
	
	tmSph.visual.vertex_colors= colours
	mesh_rd2 = pyrender.Mesh.from_trimesh(tmSph)
	show_mesh_gui([mesh_rd2])
	
	tm.export('/Users/romywilliamson/Desktop/coloured_surface.ply') ############### export as a coloured ply file
	tmSph.export('/Users/romywilliamson/Desktop/coloured_sphere.ply') ############### export as a coloured ply file

	
	
else:
	dense_tm.visual.vertex_colors= colours
	mesh_rd1 = pyrender.Mesh.from_trimesh(dense_tm)
	show_mesh_gui([mesh_rd1])
	
	dense_tm.export('/Users/romywilliamson/Desktop/coloured_surface.ply') ############### export as a coloured ply file
	












