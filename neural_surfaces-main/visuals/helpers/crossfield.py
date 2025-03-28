import sys,os

from runners import MainRunner

from mains.experiment_configurator import ExperimentConfigurator

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
import numpy as np


def GFG(arr,prec):
    new = np.array_str(arr, precision=prec, suppress_small=True)
    return new
    
def float_to_str(a_float):
	new = "{:.16f}".format(a_float)
	return new
    
def add_object(V, F, filename, mtl_filename, name, RGB, vtxCount=None):
	###### write all vertices
	###### write all faces
	###### increment vtxCount
	
	with open(mtl_filename,'a') as mtl_file:
		mtl_file.write('newmtl Material'+name+'\n')
		mtl_file.write('Ns 250.000000\n')
		mtl_file.write('Ka 1.000000 1.000000 1.000000\n')
		mtl_file.write('Kd '+str(RGB[0])+' '+str(RGB[1])+' '+str(RGB[2])+'\n')
		mtl_file.write('Ks 0.500000 0.500000 0.500000\n')
		mtl_file.write('Ke 0.000000 0.000000 0.000000\n')
		mtl_file.write('Ni 1.450000\n')
		mtl_file.write('d 1.000000\n')
		mtl_file.write('illum 2\n')

	with open(filename, 'a') as out_file:
		out_file.write('o ' + name + '\n')
		for vertex in list(V):
			out_file.write('v '+float_to_str(vertex[0])+' '+float_to_str(vertex[1])+' '+float_to_str(vertex[2])+'\n')
		out_file.write('usemtl Material'+name+'\n')
		for face in list(F):
			out_file.write('f '+str(face[0]+1+vtxCount)+
			' '+str(face[1]+1+vtxCount) +' '+str(face[2]+1+vtxCount)+'\n')
			
	vtxCount += V.shape[0]
	return vtxCount

def writequadsE(V1, F1, dir1, dir2, offset_factor, all_normals, arrow_length, filename = 'crossfield.obj', which=['A','B','C','D']):

	
	mtl_filename = filename[:-4]+'.mtl'
	path = '../data/visualisation/'
	file = open(path+filename,"w")
	file.close()
	file = open(path+mtl_filename,"w")
	file.close()
	
	with open(path+filename, 'w') as out_file:
		out_file.write('mtllib '+mtl_filename+'\n')
		
	
	vtxCount = 0
	
	
	
	
	V = np.zeros((V1.shape[0]*3, 3))
	F = np.zeros((V1.shape[0], 3), dtype='int32')
    
	for i in range(V1.shape[0]):
		V[3*i,:] = V1[i,:] 
		V[3*i+1,:] = V1[i,:] + arrow_length*dir1[i,:]
		V[3*i+2,:] = V1[i,:] + arrow_length*dir2[i,:] 
        
		F[i,:] = np.array([3*i,3*i+1,3*i+2])
	
	if 'A' in which:
		vtxCount = add_object(V, F, path+filename, path+mtl_filename, 'A', [0.0,0.0,1.0], vtxCount=vtxCount)
	
	
	V = np.zeros((V1.shape[0]*3, 3))
	F = np.zeros((V1.shape[0], 3), dtype='int32')
    
	for i in range(V1.shape[0]):
		V[3*i,:] = V1[i,:]
		V[3*i+1,:] = V1[i,:] + arrow_length*dir2[i,:]
		V[3*i+2,:] = V1[i,:] - arrow_length*dir1[i,:] 
        
		F[i,:] = np.array([3*i,3*i+1,3*i+2])
	
	if 'B' in which:
		vtxCount = add_object(V, F, path+filename, path+mtl_filename, 'B', [0.0,1.0,0.0], vtxCount=vtxCount)
	
	
	for i in range(V1.shape[0]):
		V[3*i,:] = V1[i,:] 
		V[3*i+1,:] = V1[i,:] - arrow_length*dir1[i,:]
		V[3*i+2,:] = V1[i,:] - arrow_length*dir2[i,:] 
        
		F[i,:] = np.array([3*i,3*i+1,3*i+2])
		
	if 'C' in which:
		vtxCount = add_object(V, F, path+filename, path+mtl_filename, 'C', [1.0,0.5,0.0], vtxCount=vtxCount)
	
	
	for i in range(V1.shape[0]):
		V[3*i,:] = V1[i,:] 
		V[3*i+1,:] = V1[i,:] - arrow_length*dir2[i,:]
		V[3*i+2,:] = V1[i,:] + arrow_length*dir1[i,:] 
        
		F[i,:] = np.array([3*i,3*i+1,3*i+2])
	
	if 'D' in which:
		vtxCount = add_object(V, F, path+filename, path+mtl_filename, 'D', [1.0,0.0,0.0], vtxCount=vtxCount)
	
	vtxCount = add_object(V1, F1, path+filename, path+mtl_filename, '0', [0.6,0.0,0.6], vtxCount=vtxCount)
	
	

def writequadsF(V0, dir1, dir2, offset_factor, all_normals, arrow_length, ratio, overlap=False, filename = 'crossfield.obj', which=['A','B','C', 'D']):
	
	mtl_filename = filename[:-4]+'.mtl'
	path = '../data/visualisation/'
	file = open(path+filename,"w")
	file.close()
	file = open(path+mtl_filename,"w")
	file.close()
	
	with open(path+filename, 'w') as out_file:
		out_file.write('mtllib '+mtl_filename+'\n')
		
	
	vtxCount = 0
	
	
	V1 = V0.copy() + offset_factor*all_normals
	
	V = np.zeros((V1.shape[0]*4, 3))
	F = np.zeros((V1.shape[0]*2, 3), dtype='int32')
    
	for i in range(V1.shape[0]):
		V[4*i,:] = V1[i,:] 
		
		V[4*i+1,:] = V1[i,:] + arrow_length*dir1[i,:] + arrow_length*dir2[i,:]
		V[4*i+2,:] = V1[i,:] + arrow_length*dir1[i,:] - arrow_length*dir2[i,:]
		
		V[4*i+3,:] = V1[i,:] + ratio*arrow_length*dir1[i,:] 
		
		if overlap==True:
			V[4*i:4*i+4,:] -= arrow_length*dir1[i,:]
        
		F[2*i,:] = np.array([4*i+1,4*i,4*i+2])
		F[2*i+1,:] = np.array([4*i+1,4*i+2, 4*i+3])
		
	if 'A' in which:
		vtxCount = add_object(V, F, path+filename, path+mtl_filename, 'A', [0.0,0.0,1.0], vtxCount=vtxCount)

    
	for i in range(V1.shape[0]):
		V[4*i,:] = V1[i,:] 
		
			
		V[4*i+1,:] = V1[i,:] + arrow_length*dir1[i,:] + arrow_length*dir2[i,:]
		V[4*i+2,:] = V1[i,:] - arrow_length*dir1[i,:] + arrow_length*dir2[i,:]
		
		V[4*i+3,:] = V1[i,:] + ratio*arrow_length*dir2[i,:] 
		
		if overlap==True:
			V[4*i:4*i+4,:] -= arrow_length*dir2[i,:]
        
		F[2*i,:] = np.array([4*i,4*i+1,4*i+2])
		F[2*i+1,:] = np.array([4*i+2,4*i+1, 4*i+3])
	
	if 'B' in which:
		vtxCount = add_object(V, F, path+filename, path+mtl_filename, 'B', [1.0,0.0,0.0], vtxCount=vtxCount)
	
	
	for i in range(V1.shape[0]):
		V[4*i,:] = V1[i,:] 
		
		V[4*i+1,:] = V1[i,:] - arrow_length*dir1[i,:] - arrow_length*dir2[i,:] 
		V[4*i+2,:] = V1[i,:] - arrow_length*dir1[i,:] + arrow_length*dir2[i,:]
		
		V[4*i+3,:] = V1[i,:] - ratio*arrow_length*dir1[i,:] 
		
		if overlap==True:
			V[4*i:4*i+4,:] += arrow_length*dir1[i,:]
        
		F[2*i,:] = np.array([4*i+1,4*i,4*i+2])
		F[2*i+1,:] = np.array([4*i+1,4*i+2, 4*i+3])
	
	if 'C' in which:
		vtxCount = add_object(V, F, path+filename, path+mtl_filename, 'C', [0.0,0.0,1.0], vtxCount=vtxCount)
	
	
	for i in range(V1.shape[0]):
		V[4*i,:] = V1[i,:] 
		
		V[4*i+1,:] = V1[i,:] - arrow_length*dir1[i,:] - arrow_length*dir2[i,:]
		V[4*i+2,:] = V1[i,:] + arrow_length*dir1[i,:] - arrow_length*dir2[i,:]
		
		V[4*i+3,:] = V1[i,:] - ratio*arrow_length*dir2[i,:]
		
		if overlap==True:
			V[4*i:4*i+4,:] += arrow_length*dir2[i,:] 
        
		F[2*i,:] = np.array([4*i,4*i+1,4*i+2])
		F[2*i+1,:] = np.array([4*i+2,4*i+1, 4*i+3])
		
	if 'D' in which:
		vtxCount = add_object(V, F, path+filename, path+mtl_filename, 'D', [1.0,0.0,0.0], vtxCount=vtxCount)
	
	#vtxCount = add_object(V0, F0, path+filename, path+mtl_filename, '0', [0.6,0.6,1.0], vtxCount=vtxCount)	

	return

    
    
    
