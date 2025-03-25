import sys,os



import numpy as np
import trimesh#to load mesh
import igl

def normalise_mesh(name, normalisation='A', rotate=False, directory='../data/'): ### e.g. name='HUMAN24461' (name of obj file in data folder)

	print(directory)
	mesh_path = directory + name+'.obj'        #################### comment out if you want to directly specify mesh path
	print(mesh_path)
	tm1 = trimesh.load(mesh_path)#load mesh
	V =tm1.vertices
	F = tm1.faces
	
	#translation = - np.mean(V,axis=0)
	translation = 0.0
	V = V + translation
	
	##### for testing only #######
	if rotate==True:
		pi = np.pi
		t = pi/3.0
		M = np.array([[1.0, 0.0, 0.0],
			      [0.0, np.cos(t), np.sin(t)],
			      [0.0, -1.0*np.sin(t), np.cos(t)]])
		V = V@M
	##############################
	
	

	if normalisation=='B':
		
		max_disp = np.max(abs(V)) ## maximum distance out from origin in x,y and z directions
		sf = 1/max_disp
		V1 = V*sf
		
		
		
		igl.write_triangle_mesh(directory + name+'_nB.obj',V1,F)

	elif normalisation=='A':
		areas = tm1.area_faces
		total_area = np.sum(areas)
		print('total area',total_area)
		sf = np.sqrt(4.0*np.pi)/ np.sqrt(total_area)
		V2 = sf * V 
		
		
		
		igl.write_triangle_mesh(directory + name+'_nA'+'.obj',V2,F)
		
	print('scale factor',sf)
	print('translation', translation)
	return sf, translation
	
	




