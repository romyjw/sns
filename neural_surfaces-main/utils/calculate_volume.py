import torch
def calculate_volume(sphere_vertices, sphere_faces):
	import torch
	A = sphere_vertices[sphere_faces[:,0]]
	B = sphere_vertices[sphere_faces[:,1]]
	C = sphere_vertices[sphere_faces[:,2]]
	
	vol_mat = torch.stack((B-A, C-A, A)).permute(1,2,0)
	face_volumes = vol_mat.det()
	return face_volumes.sum()/6.0