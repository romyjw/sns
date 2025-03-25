import torch
import numpy as np
import trimesh
import igl
import numpy as np
import matplotlib.pyplot as plt

def compute_rotation_sig(S_corresp, special_index = 0):
	PI = torch.pi
	a = special_index

	theta = torch.acos(S_corresp[a,2])
	phi = torch.atan2(S_corresp[a,1], S_corresp[a,0])
	

	Rz = torch.tensor([[torch.cos(phi), -torch.sin(phi), 0.0],
				[torch.sin(phi), torch.cos(phi), 0.0],
				[0.0 , 0.0, 1.0]]).T
	
	Ry = torch.tensor([[torch.cos(+ theta), 0.0, -torch.sin( + theta)],
				[0.0, 1.0, 0.0],
				[torch.sin( theta), 0.0, torch.cos( + theta)]])

	R = Ry @ Rz
	#print(R.shape)
	return R

def rotate2pole1(P, rotation_sig):### rotate the ldmk to north pole

	R = rotation_sig
	#print('rot matrix shape 1', R.shape)

	return (P @ R.T)
	
def rotate2pole2(P, rotation_sig):  ###### rotate the ldmk back from the north pole
	
	R = rotation_sig
	#print('rot matrix shape 2', R.shape)

	return (P @ R)
	

def compute_rotation2D(lands_source, lands_target, rotation_sig1, rotation_sig2, special_index = 0):

	lands_source_NA = rotate2pole1(lands_source, rotation_sig1)	### NA means 'north-aligned'
	lands_target_NA = rotate2pole1(lands_target, rotation_sig2)	
	
	a = special_index
	if lands_source.shape[0] == 1: #if only one landmark, don't do any 2D rotation
		return torch.eye(3)
		
	elif lands_source.shape[0] == 2: #if there are exactly two landmarks, find 2D rotation that aligns 2nd landmark as well
		theta1 = torch.atan2(lands_source_NA[a+1, 1], lands_source_NA[a+1, 0] )
		theta2 = torch.atan2(  lands_target_NA[a+1, 1], lands_target_NA[a+1, 0]  )
		theta = theta2 - theta1
		
		R = torch.tensor([[torch.cos(theta), -1.0*torch.sin(theta), 0.0],
				[torch.sin(theta), torch.cos(theta), 0.0],
				[0.0, 0.0, 1.0]])
		return R.T		
	
	elif lands_source.shape[0] >= 3: #if there are exactly two landmarks, find 2D rotation that aligns 2nd landmark as well
		theta1 = torch.atan2(lands_source_NA[a+1, 1], lands_source_NA[a+1, 0] )
		theta2 = torch.atan2(  lands_target_NA[a+1, 1], lands_target_NA[a+1, 0]  )
		theta = theta2 - theta1
		
		R = torch.tensor([[torch.cos(theta), -1.0*torch.sin(theta), 0.0],
				[torch.sin(theta), torch.cos(theta), 0.0],
				[0.0, 0.0, 1.0]])
		return R.T	
		
	
	
	
	###### this isn't working
	elif lands_source.shape[0] >= 3:
		
		lands_source_NA_2D = lands_source_NA[1:, :-1] ###### this assumes that special index is 0. 
		lands_target_NA_2D = lands_target_NA[1:, :-1]
		
		center_lands_source = lands_source_NA_2D - lands_source_NA_2D.mean(dim=0)
		center_lands_target = lands_target_NA_2D - lands_target_NA_2D.mean(dim=0)
        	
		H = center_lands_source.transpose(0,1).matmul(center_lands_target)
		u, e, v = torch.svd(H)
		R = v.matmul(u.transpose(0,1)).detach()

		# check rotation is not a reflection
		if R.det() < 0.0:
			v[:, -1] *= -1
			R = v.matmul(u.transpose(0,1)).detach()
			
		big_R = torch.zeros((3,3))
		big_R[:2,:2] = R 
		big_R[2,2] = 1.0
		return big_R

def test1():

	corresp = torch.randn((2,3))
	corresp = (corresp.T/torch.sqrt(torch.sum(corresp**2, axis = 1))).T
	
	print(corresp)
	rotation_sig = compute_rotation_sig(corresp, special_index = 0)
	
	print(rotate2pole1(corresp, rotation_sig))
	
	
	return
	
def test2():
	PI = torch.pi
	
	src_points = torch.randn((2,3))
	src_points = (src_points.T/torch.sqrt(torch.sum(src_points**2, axis = 1))).T
	
	a = 2.0*PI * torch.rand(1)
	b = 2.0*PI * torch.rand(1)
	c = 2.0*PI * torch.rand(1)
	
	Rx = torch.tensor([[1.0, 0.0, 0.0],
				[0.0, torch.cos(a), -torch.sin(a)],
				[0.0, torch.sin(a), torch.cos(a)]])
	
	Ry = torch.tensor([[torch.cos(b), 0.0, -torch.sin(b)],
				[0.0, 1.0, 0.0],
				[torch.sin(b), 0.0, torch.cos(b)]])
	
	Rz = torch.tensor([[torch.cos(c), torch.sin(c), 0.0],
				[-torch.sin(c), torch.cos(c), 0.0],
				[0.0, 0.0, 1.0]])
				
	tgt_points = src_points @ Rx @ Ry @ Rz
	
	
	
	
	
	rotation_sig1 = compute_rotation_sig(src_points)
	rotation_sig2 = compute_rotation_sig(tgt_points)
	
	src_NA = rotate2pole1(src_points, rotation_sig1)
	tgt_NA = rotate2pole1(tgt_points, rotation_sig2)
	
	print('src na',src_NA)
	print('tgt na', tgt_NA)
	
	rotation2D = compute_rotation2D(src_NA, tgt_NA, rotation_sig1, rotation_sig2)
	
	temp = src_NA @ rotation2D
	print('x', temp[:,0] - tgt_NA[:,0])
	print('y', temp[:,1] - tgt_NA[:,1])
	print('z', temp[:,2] - tgt_NA[:,2])
	#plt.show()
	
	
	
test2()

#### test1()
