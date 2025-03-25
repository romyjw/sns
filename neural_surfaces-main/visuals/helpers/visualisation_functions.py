import sys,os

import pyglet
pyglet.options['shadow_window'] = True

import pyrender#to display mesh
import numpy as np
import trimesh#to load mesh

import matplotlib
import matplotlib.pyplot as plt
import igl


try:
    from sklearn.neighbors import KDTree
except:
    pass

def scene_factory(render_list, return_nodes=False):
    
    scene = pyrender.Scene(ambient_light=0.5*np.array([1.0, 1.0, 1.0, 1.0]))
    nd_list=[]
    for m in render_list:
        nd=scene.add(m)
        nd_list.append(nd)
    
    if return_nodes:
        return scene, nd_list
    else:
        return scene

def show_mesh_gui(rdobj):
    scene = scene_factory(rdobj)
    v=pyrender.Viewer(scene, use_raymond_lighting=True,show_world_axes=True)
   
    del v

def render_trimesh(tm, colors=None):
    if colors is not None:
        tm.visual.vertex_colors = colors
    mesh_rd = pyrender.Mesh.from_trimesh(tm)
    show_mesh_gui([mesh_rd])

def render_trimeshes(tms, colorslist=None):
    mesh_rds=[]
    
    for i in range(len(tms)):
        
        mesh_rds.append (pyrender.Mesh.from_trimesh(tms[i]))
        if colorslist is not None:
             tms[i].visual.vertex_colors = colorslist[i]
             
             
    show_mesh_gui(mesh_rds)		
		
		

def run_gui_edges(edge_rdobj):
    scene = scene_factory(edge_rdobj) 
    v=pyrender.Viewer(scene, use_raymond_lighting=True)
 
    

def make_quadsA(V1, dir1, dir2, offset_factor, all_normals, arrow_length):
    V = np.zeros((V1.shape[0]*4, 3))
    F = np.zeros((V1.shape[0]*2, 3), dtype='int32')
    for i in range(V1.shape[0]):
        V[4*i,:] = V1[i,:]
        V[4*i+1:] = V1[i,:] + arrow_length*dir1[i,:]
        V[4*i+2:] = V1[i,:] + arrow_length*dir1[i,:] + arrow_length*dir2[i,:]
        V[4*i+3:] = V1[i,:] + arrow_length*dir2[i,:]
        
        F[2*i,:] = np.array([4*i,4*i+1,4*i+3])
        F[2*i+1,:] = np.array([4*i+2,4*i+3,4*i+1])
        
    return V,F

def make_quadsB(V1, dir1, dir2, offset_factor, all_normals, arrow_length):
    V = np.zeros((V1.shape[0]*5, 3))
    F = np.zeros((V1.shape[0]*4, 3), dtype='int32')
    for i in range(V1.shape[0]):
        V[5*i,:] = V1[i,:]
        V[5*i+1:] = V1[i,:] + arrow_length*dir1[i,:]
        V[5*i+2:] = V1[i,:] + arrow_length*dir2[i,:]
        V[5*i+3:] = V1[i,:] - arrow_length*dir1[i,:]
        V[5*i+4:] = V1[i,:] - arrow_length*dir2[i,:]
        
        F[4*i,:] = np.array([5*i,5*i+1,5*i+2])
        F[4*i+1,:] = np.array([5*i,5*i+2,5*i+3])
        F[4*i+2,:] = np.array([5*i,5*i+3,5*i+4])
        F[4*i+3,:] = np.array([5*i,5*i+4,5*i+1])
        
    return V,F
    
def make_quadsC(V1, dir1, dir2, offset_factor, all_normals, arrow_length):
    V = np.zeros((V1.shape[0]*12, 3))
    F = np.zeros((V1.shape[0]*4, 3), dtype='int32')
    
    for i in range(V1.shape[0]):
        V[12*i,:] = V1[i,:] + 0.01*(dir1[i,:] + dir2[i,:])
        V[12*i+1,:] = V1[i,:] + arrow_length*dir1[i,:] - 0.001
        V[12*i+2,:] = V1[i,:] + arrow_length*dir2[i,:] + 10.0
        
        V[12*i+3,:] = V1[i,:] + 0.0001
        V[12*i+4,:] = V1[i,:] + arrow_length*dir2[i,:] + 0.00001
        V[12*i+5,:] = V1[i,:] - arrow_length*dir1[i,:] + 0.001
        
        V[12*i+6,:] = V1[i,:]
        V[12*i+7,:] = V1[i,:] - arrow_length*dir1[i,:]
        V[12*i+8,:] = V1[i,:] - arrow_length*dir2[i,:]
        
        V[12*i+9,:] = V1[i,:] - 0.0001
        V[12*i+10,:] = V1[i,:] - arrow_length*dir2[i,:] - 0.01
        V[12*i+11,:] = V1[i,:] + arrow_length*dir1[i,:] - 0.000001
        
        F[4*i,:] = np.array([12*i,12*i+1,12*i+2])
        F[4*i+1,:] = np.array([12*i+3,12*i+4,12*i+5])
        F[4*i+2,:] = np.array([12*i+6,12*i+7,12*i+8])
        F[4*i+3,:] = np.array([12*i+9,12*i+10,12*i+11])
        
    return V,F
    

def make_quadsD(V1, dir1, dir2, offset_factor, all_normals, arrow_length):
    V = np.zeros((V1.shape[0]*12, 3))
    F = np.zeros((V1.shape[0]*4, 3), dtype='int32')
    
    for i in range(V1.shape[0]):
        V[12*i,:] = V1[i,:] + all_normals[i,:]*offset_factor
        V[12*i+1,:] = V1[i,:] + arrow_length*dir1[i,:]+ all_normals[i,:]*offset_factor
        V[12*i+2,:] = V1[i,:] + arrow_length*dir2[i,:] + all_normals[i,:]*offset_factor
        
        V[12*i+3,:] = V1[i,:] + all_normals[i,:]*offset_factor
        V[12*i+4,:] = V1[i,:] + arrow_length*dir2[i,:] + all_normals[i,:]*offset_factor
        V[12*i+5,:] = V1[i,:] - arrow_length*dir1[i,:]+ all_normals[i,:]*offset_factor
        
        V[12*i+6,:] = V1[i,:]+ all_normals[i,:]*offset_factor
        V[12*i+7,:] = V1[i,:] - arrow_length*dir1[i,:]+ all_normals[i,:]*offset_factor
        V[12*i+8,:] = V1[i,:] - arrow_length*dir2[i,:]+ all_normals[i,:]*offset_factor
        
        V[12*i+9,:] = V1[i,:]+ all_normals[i,:]*offset_factor
        V[12*i+10,:] = V1[i,:] - arrow_length*dir2[i,:] + all_normals[i,:]*offset_factor
        V[12*i+11,:] = V1[i,:] + arrow_length*dir1[i,:] + all_normals[i,:]*offset_factor
        
    
    rds = []
    for i in range(0, V.shape[0], 3):
    	tm = trimesh.Trimesh(vertices=V[i:i+3, :], faces=np.array([[0,1,2]]))
    	
    	tm.visual.vertex_colors=np.array([[1.0*(i%2),0.0,0.0],[1.0*(i%2),0.0,0.0],[1.0*(i%2),0.0,0.0]])
    	rds.append(pyrender.Mesh.from_trimesh(tm))
        
    return rds




