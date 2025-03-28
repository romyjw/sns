import sys,os

import pyglet
pyglet.options['shadow_window'] = True

import pyrender#to display mesh
import numpy as np
import trimesh#to load mesh
import igl

import matplotlib
import matplotlib.pyplot as plt

from scipy import sparse
from scipy import linalg

from sklearn.neighbors import KDTree
from visuals.helpers.visualisation_functions import *

import pickle

import warnings
from utils.normalise_mesh import normalise_mesh


#warnings.simplefilter(action='ignore', category=RuntimeWarning)



namespace = sys.argv[-2]
level=sys.argv[-1]
base_mesh_path = '../data/analytic/sphere/sphere'+str(level)+'.obj'
path = '../data/analytic/'+namespace
N=100000

#formula='P *  np.transpose(( np.transpose(np.ones_like(P))  +   0.3 * (0.4*((U*(np.pi-U))**2)*np.cos(18*U)**2 + 0.6*((U*(np.pi-U))**2)*np.sin(6*V +     18*U)**2) ))'


with open(path+'/formula.txt') as formulafile:
	formula = formulafile.read()



make_ptcloud=False
write_ptcloud=False
make_mesh=True

if not (os.path.isdir(path)):
	    os.mkdir(path)




def gen_pointcloud(N, formula):
    samples = np.random.randn(3,N)
    samples = samples/np.sqrt(np.einsum('ij,ij->j',samples,samples))
    samples = np.transpose(samples)#now 2000 x 3

    U = np.arccos(samples[:,2])###### arccos(z)
    V = np.arctan(samples[:,1]/samples[:,0])##################### arctan (y/x)    ### would be easier to use arctan2(y,x).
    V = V + (samples[:,1]/np.sin(U) < 0)*np.pi
    
    P = samples
    
    
    R = np.zeros_like(P)
    R = eval(formula)
    #print(R)


    ###############colours######################################


    normals = PCA_normal_estimation(R, 3)# (expensive)
    colors = (normals.copy() + 1)*0.5



    ptcloud_mesh = pyrender.Mesh.from_points(R, colors=colors)
    
    show_mesh_gui([ptcloud_mesh])#Display meshes.

    return R,normals


def gen_mesh(base_mesh_path,formula,level):
	tm = trimesh.load(base_mesh_path)
	P = tm.vertices
	F = tm.faces
	
	U = np.arccos(P[:,2])###### arccos(z)
	
	V = np.arctan(P[:,1]/P[:,0])#####################
	
	
	V = V + (P[:,1]/np.sin(U) < 0)*np.pi
	
	V[np.isnan(V)] = 0

	#P = np.transpose(np.stack((np.sin(U)*np.cos(V),   np.sin(U)*np.sin(V), np.cos(U))))  ## check
	
	
	V = eval(formula)
	
	#normals = PCA_normal_estimation(V, 20)# (expensive)
	colors = np.array([0.4, 0.6, 0.8])#0.5*normals.copy()+ 0.5
    
	ret = igl.write_triangle_mesh(str('../data/analytic/'+namespace+'/mesh'+str(level)+'.obj'), V, F)
	#np.save(str('../data/analytic/'+namespace+'/mesh'+str(level)+'_normals.npy'),normals)
	
	tm2 = trimesh.load(str('../data/analytic/'+namespace+'/mesh'+str(level)+'.obj'))
	
	tm2.visual.vertex_colors = colors
	
	mesh = pyrender.Mesh.from_trimesh(tm2)

	show_mesh_gui([mesh])#Display meshes.
	
	return
	
	
    
if make_mesh==True:
	gen_mesh(base_mesh_path, formula, level)
	
	sf, translation = normalise_mesh('mesh'+str(level), normalisation='B', rotate=False, directory='../data/analytic/'+namespace+'/')
	sf, translation = normalise_mesh('mesh'+str(level), normalisation='A', rotate=False, directory='../data/analytic/'+namespace+'/')
	
if make_ptcloud==True:
	R,normals = gen_pointcloud(N,formula)


   

if write_ptcloud==True:
	

	np.save(str('../data/analytic/'+namespace+'/ptcloud'+str(N)+'.npy'),R)
	np.save(str('../data/analytic/'+namespace+'/ptcloud'+str(N)+'_normals.npy'),normals)

	#file = open(str('../data/analytic/'+namespace+'/formula.txt'),'w')
	file.write(formula)
	file.close()


