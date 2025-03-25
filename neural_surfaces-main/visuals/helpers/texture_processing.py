############### Take a texture image that has been created by painting onto a mesh with my spherical polar-based UVs, and tesselate it so it may be used as a texture image for a different
############### shape that has spherical polar-based UVs but a different triangulation.

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import cv2
import torch

def write_transferred_texture(textured_obj ,textured_sphere_vertices ,
							sphere_vertices , geom_vertices , geom_faces, neural_image_vertices, src_vertices):
																
	_, T_texture_vertices, _ = parse_obj(textured_obj)
	
	print('t texture vertices',len(T_texture_vertices)) ### coords of texture vertices in 2D of Max Planck
	import trimesh
	textured_shape = trimesh.load(textured_obj)
	textured_faces = textured_shape.faces
	
	print('doing bary calcualtion')
	bary_info = find_bary(textured_sphere_vertices, sphere_vertices, textured_faces, 'bary.txt', neural_image_vertices, src_vertices)
	print('finished bary calcualtion')	
		
	new_texture_vertices = []
		
	for i in range(len(bary_info)):
		face_id, a, b, c = bary_info[i]
		#a = 0.33333333333333
		#b=0.3333333333333333
		#c=0.3333333333333333
				
		vtx_index1 = textured_faces[face_id][0]
		vtx_index2 = textured_faces[face_id][1]
		vtx_index3 = textured_faces[face_id][2]
		
		new_texture_vertices.append(     (a*float(T_texture_vertices[vtx_index1][0]) + b*float(T_texture_vertices[vtx_index2][0]) + c*float(T_texture_vertices[vtx_index3][0]),
										a*float(T_texture_vertices[vtx_index1][1]) + b*float(T_texture_vertices[vtx_index2][1]) + c*float(T_texture_vertices[vtx_index3][1]) ))
	
	
	with open('../data/visualisation/textured_from_map.obj', 'w') as out_file:
		for vertex in list(geom_vertices):
			out_file.write('v '+str(float(vertex[0]))+' '+str(float(vertex[1]))+' '+str(float(vertex[2]))+'\n')
		for texcoord in new_texture_vertices:
			out_file.write('vt '+str(float(texcoord[0] ))+' '+str(float(texcoord[1]) )+'\n')
		for face in geom_faces:
			out_file.write('f '+str(face[0]+1)+'/'+str(face[0]+1 )+' '+str(face[1]+1)+'/'+str(face[1]+1)+' '+str(face[2]+1)+'/'+str(face[2]+1)+'\n')


def parse_obj(filename):
	vertices = []
	texture_vertices = []
	faces = []

	# Open the OBJ file for reading
	with open(filename, 'r') as obj_file:
		for line in obj_file:
			tokens = line.strip().split()

			if not tokens:
		    		continue

			# Check the type of line: vertex, texture vertex, or face
			if tokens[0] == 'v':
		    		# Geometric vertex
				vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
			elif tokens[0] == 'vt':
				# Texture vertex
				texture_vertices.append([float(tokens[1]), float(tokens[2])])
			elif tokens[0] == 'f':
				# Face
				face = []
				for token in tokens[1:]:
					vertex_info = token.split('/')
					vertex_index = int(vertex_info[0])
					texture_index = int(vertex_info[1]) if len(vertex_info) > 1 and vertex_info[1] else None
					face.append((vertex_index, texture_index))
				faces.append(face)

	# Print the extracted information
	#print("Geometric Vertices:")
	#for vertex in vertices:
	#    print(vertex)

	#print("\nTexture Vertices:")
	#for texture_vertex in texture_vertices:
	#    print(texture_vertex)

	#print("\nFaces:")
	#for face in faces:
	#    print(face)
	
	return vertices, texture_vertices, faces
	



def spherical_angle(a,b,c):
	epsilon=1e-8 
	num = torch.sum(b*c) - torch.sum(c*a)*torch.sum(a*b)
	
	denom = torch.sqrt(torch.sum(torch.linalg.cross(c,a)**2)) * torch.sqrt(torch.sum(torch.linalg.cross(a,b)**2))
	out = torch.acos(torch.clamp(num/denom, -1 + epsilon, 1 - epsilon))
	return out

def area(a,b,c):
	
	v1,v2 = b-a, c-a
	cross = torch.linalg.cross(v1,v2)
	
	return 0.5* torch.sqrt(torch.sum(cross**2))


def find_bary(src_sphere_vertices, tgt_sphere_vertices, src_sphere_faces, bary_filename, neural_image_vertices, src_vertices):
	src_sphere_vertices = torch.tensor(src_sphere_vertices, dtype=torch.float64)
	tgt_sphere_vertices = torch.tensor(tgt_sphere_vertices, dtype=torch.float64)
	
	src_sphere_vertices = (src_sphere_vertices.T/(torch.sqrt(torch.sum(src_sphere_vertices**2, axis=1)).T)).T
	tgt_sphere_vertices = (tgt_sphere_vertices.T/(torch.sqrt(torch.sum(tgt_sphere_vertices**2, axis=1)).T)).T
	
	
	A = torch.tensor(src_sphere_vertices[src_sphere_faces[:,0]])
	B = torch.tensor(src_sphere_vertices[src_sphere_faces[:,1]])
	C = torch.tensor(src_sphere_vertices[src_sphere_faces[:,2]])
	
	A_shape = torch.tensor(src_vertices[src_sphere_faces[:,0]])
	B_shape = torch.tensor(src_vertices[src_sphere_faces[:,1]])
	C_shape = torch.tensor(src_vertices[src_sphere_faces[:,2]])

    
	output=[]
	with open(bary_filename,'w') as bary_file:
		for i in range (tgt_sphere_vertices.shape[0]):
    		
			tgt_vertex = torch.tensor(tgt_sphere_vertices[i,:])
			tgt_neural_image_vertex = torch.tensor(neural_image_vertices[i,:])
			        
        
			tet1 = torch.stack([tgt_vertex.unsqueeze(0).repeat(A.shape[0], 1), C, A]).permute((1,0,2))
			vol1 =    torch.linalg.det(tet1)                #a.(b x c) or det ([a , b, c])
                
			tet2 = torch.stack([tgt_vertex.unsqueeze(0).repeat(A.shape[0], 1), A, B]).permute((1,0,2))
			vol2 =    torch.linalg.det(tet2)
        
			tet3 = torch.stack([tgt_vertex.unsqueeze(0).repeat(A.shape[0], 1), B,C]).permute((1,0,2))
			vol3 =    torch.linalg.det(tet3)
        
			correct_face = (          torch.logical_and (               torch.logical_and((vol1>0) , (vol2>0) )  , (vol3>0) ))
        	
			face_id = correct_face.nonzero()[0][0]
			if correct_face.nonzero().shape[0]!=1 or correct_face.nonzero().shape[1]!=1:
				raise ValueError('error')
				
			normal = torch.linalg.cross(C_shape[face_id,:] - A_shape[face_id,:], B_shape[face_id,:] - A_shape[face_id,:])
			normal = normal/torch.sqrt(torch.sum(normal**2))
			
			tgt_neural_image_vertex = tgt_neural_image_vertex - normal*  ( torch.sum(tgt_neural_image_vertex*normal) - torch.sum(A_shape[face_id,:]*normal))
        	
			bary1 = area(tgt_neural_image_vertex, B_shape[face_id,:], C_shape[face_id,:]) 
			bary2 = area(tgt_neural_image_vertex, A_shape[face_id,:], C_shape[face_id,:])
			bary3 = area(tgt_neural_image_vertex, A_shape[face_id,:], B_shape[face_id,:]) 
			
			total = bary1 + bary2 + bary3
		        	
			bary = (bary1.squeeze()/total, bary2.squeeze()/total, bary3.squeeze()/total)
			        	
			if i!=0:
				bary_file.write('\n')
			bary_file.write(str(int(face_id))+' '+str(float((bary[0])))+' '+str(float((bary[1])))+' '+str(float((bary[2]))))
			output.append((int(face_id), float(bary[0]), float(bary[1]), float(bary[2])))
			
		return output
	
def copy_over_texture(mesh_filename, textured_mesh_filename):

	with open(mesh_filename, 'r') as geom_file:
		geom_lines = geom_file.readlines()
	with open(textured_mesh_filename, 'r') as tex_file:
		tex_lines = tex_file.readlines()
	
	print('have read files')

	with open('../data/'+mesh_filename+'_textured.obj', 'w') as out_file:
		for line in geom_lines:
			if line[:2]=='v ':
				out_file.write(line)
		print('wrote vertices')
		for line in tex_lines:
			if line[:2]=='vt' or line[:2]=='f ':
				out_file.write(line)
		print('wrote texture')
        	
	
	
def write_projection_texture(geom_vertices, faces, filename, side='xy', magnification = 4.0):
	########################find new texture coordinates###################
	scale = 1.0/magnification
	texcoords = []
	for i in range(geom_vertices.shape[0]):
		x = geom_vertices[i,0]
		y = geom_vertices[i,1]
		z = geom_vertices[i,2]
		if side=='xy':
			texcoords.append((scale*x,scale*y))
		elif side=='smvat_shrec':
			texcoords.append((scale*z,scale*y))
		elif side=='xz':
			texcoords.append((1.0 - scale*x,scale*z))
		elif side == 'yz':
			texcoords.append((scale*y,scale*z))
		    
	#####################now write new obj file##########################################

	with open(filename, 'w') as out_file:
		for vertex in list(geom_vertices):
			out_file.write('v '+str(float(vertex[0]))+' '+str(float(vertex[1]))+' '+str(float(vertex[2]))+'\n')
		for texcoord in texcoords:
			out_file.write('vt '+str(float(texcoord[0] ))+' '+str(float(texcoord[1]) )+'\n')
		for face in faces:
			out_file.write('f '+str(face[0]+1)+'/'+str(face[0] + 1)+' '+str(face[1]+1)+'/'+str(face[1] + 1)+' '+str(face[2]+1)+'/'+str(face[2] + 1)+'\n')
			