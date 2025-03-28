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

from sklearn.neighbors import KDTree, NearestNeighbors
from visuals.helpers.visualisation_functions import *

import pickle

namespace = 'sphere'
path = '../analytic/'+namespace





def find_nbrs(points):#make an affinity matrix --equiv to covariance matrix. based on dot product/angle between vertices with centre.

    C = points@points.T
    print('C:',C)
    thresh_C = C > 0.4
    print(thresh_C)
    indices = []
    for i in range(12):
        indices.append([])
        for j in range(12):
            if thresh_C[i,j]==True and i!=j:
                indices[i].append(j)
        print(len(indices[i]))
    indices=np.array(indices)
    return indices,thresh_C
    
    
def refine(V,F):
    print('refining')
    f_start = F.shape[0]
    v_start = V.shape[0]
    e_start = v_start + f_start - 2 #using Euler's formula V - E + F = 2
    
    f_new = f_start*4 #each face is split into 4 faces
    v_new = v_start + e_start#every old edge contributes 1 new vertex
    
    
    new_F = np.zeros((f_new,3),dtype='int64')
    new_V = np.zeros((v_new ,3))
    #print('new-V shape',new_V.shape)
    
     
    new_V[:v_start,:] = V#copy over the old vertices
    
    midpoint_dict = {}#make dictionary to keep track of which edges have been subdivided and the index of the new vertex
    
    v_count = v_start
    f_count = f_start
    
    for i in range(f_start):
        c0 = F[i,0]
        c1 = F[i,1]
        c2 = F[i,2]
        
        v0 = V[c0,:]
        v1 = V[c1,:]
        v2 = V[c2,:]
        
        if (min(c1,c2),max(c1,c2)) in midpoint_dict:
            
            
            c3 = midpoint_dict[(min(c1,c2),max(c1,c2))]
        else:
            
            c3 = v_count
            
            v3 = (v1+v2)/2.0
            new_V[c3,:] = v3
            midpoint_dict[(min(c1,c2),max(c1,c2))] = c3
            v_count+=1#keep track of. 
            
            
        if (min(c0,c2),max(c0,c2)) in midpoint_dict:
            c4 = midpoint_dict[(min(c0,c2),max(c0,c2))]
            
        else:
            
            c4 = v_count
            
            v4 = (v0+v2)/2.0
            new_V[c4,:] = v4
            midpoint_dict[(min(c0,c2),max(c0,c2))] = c4
            v_count+=1#keep track of. 
        
        
        if (min(c0,c1),max(c0,c1)) in midpoint_dict:
            c5 = midpoint_dict[(min(c0,c1),max(c0,c1))]
            
        else:
            
            c5 = v_count
            
            v5 = (v0+v1)/2.0
            new_V[c5,:] = v5
            midpoint_dict[(min(c0,c1),max(c0,c1))] = c5
            v_count+=1#keep track of. 
  
        
        new_F[i,:] = np.array([c3,c4,c5])
        new_F[f_count,:] = np.array([c3,c2,c4])
        f_count+=1
        new_F[f_count,:] = np.array([c5,c4,c0])
        f_count+=1
        new_F[f_count,:] = np.array([c1,c3,c5])
        f_count+=1
    
    new_V = (  new_V.T / np.sqrt(np.einsum('ij,ij->i',new_V,new_V) )).T#project vertices to sphere
    return new_V,new_F


####### construct the vertices of a regular icosahdron #################
phi = (1+np.sqrt(5))/2.0
temp = np.array([[0.0, 1.0, phi],
                   [0.0, -1.0, phi],
                   [0.0, 1.0, -phi],
                   [0.0, -1.0, -phi]]).T
                   
V_init = np.zeros((3,12))
V_init[:,:4] = temp
V_init[:,4:8] = temp[(2,0,1),:]
V_init[:,8:] = temp[(1,2,0),:]

V_init=V_init.T

V_init = V_init/np.sqrt(1+phi**2)
print(V_init)

###############  find one-ring of each vertex ###########################
nbr_indices,thresh_C = find_nbrs(V_init)
print(nbr_indices)

############### construct faces of the icosahdron #######################
faces = []
for i in range(12):
    for j in range(5):
        for k in range(5):
            v1 = nbr_indices[i,j]
            v2 = nbr_indices[i,k]
            
            if (v1 > i) and (v2 > i) and (v2 in nbr_indices[v1,:]):
                face = [i,v1,v2]
                face.sort()
                
                if not face in faces:
                    faces.append(face)

print(len(faces))#there should be 20 ;) if not, there's something wrong.
print(faces)

F = np.array(faces)

for i in range(20):
    a = V_init[F[i,1]] - V_init[F[i,0]]
    b = V_init[F[i,2]] - V_init[F[i,0]]
    c = -1.0*V_init[F[i,0]]
    M = np.stack((a,b,c))
    if np.linalg.det(M)>0:
        F[i,:] = F[i,(1,0,2)]

V = V_init



if not (os.path.isdir(str('../data/analytic/sphere'))):
    os.mkdir(str('../data/analytic/sphere'))
    
    

ret = igl.write_triangle_mesh('../data/analytic/sphere/sphere0.obj', V, F)
tm = trimesh.load('../data/analytic/sphere/sphere0.obj')
mesh = pyrender.Mesh.from_trimesh(tm)

show_mesh_gui([mesh])#Display meshes.



for level in range (1,10):
    V,F = refine(V,F)
    print('V shape',V.shape)
    print('F shape',F.shape)    
    ret = igl.write_triangle_mesh('../data/analytic/sphere/sphere'+str(level)+'.obj', V, F)
    tm = trimesh.load('../data/analytic/sphere/sphere'+str(level)+'.obj')
    mesh = pyrender.Mesh.from_trimesh(tm)

    show_mesh_gui([mesh])#Display meshes.
    
    






#np.save(str('../analytic/'+namespace+'/ptcloud'+str(N)+'.npy'),R)
#np.save(str('../analytic/'+namespace+'/ptcloud'+str(N)+'_normals.npy'),normals)

#file = open(str('../analytic/'+namespace+'/formula.txt'),'w')
#file.write(formula)
#file.close()


