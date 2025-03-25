import torch
import numpy as np
import trimesh
import igl
import numpy as np


def compute_inversion_sig(S_corresp, special_indices = [0,1]): ### same for mapping to poles, and back from poles.

	a,b = special_indices[0], special_indices[1] #by default, take the first two landmarks	

	midpoint = 0.5*(S_corresp[a,:]+S_corresp[b,:])
	proj_midpoint = midpoint/torch.sum(midpoint**2)
	half_angle = torch.acos(torch.sum(S_corresp[a,:]*S_corresp[b,:]))/2.0
	
	#print('angle in degrees: ',180*half_angle/torch.pi)
	
	
	
	
	######### Use rotation - solution of Procrustes problem - as a way to align the sector with v=pi.
	######### I think this is overkill, should be able to calculate more straightforwardly.
	
	lands_source = torch.stack([proj_midpoint,
					S_corresp[a,:],
					S_corresp[b,:],
					torch.tensor([0.0,0.0,0.0])])
					
	lands_target = torch.tensor([[-1.0, 0.0, 0.0], 
					[-torch.cos(half_angle), 0.0, torch.sin(half_angle) ],
					[-torch.cos(half_angle), 0.0, -torch.sin(half_angle) ],
					[0.0, 0.0, 0.0]])
	
	center_lands_source = lands_source - lands_source.mean(dim=0)
	center_lands_target = lands_target - lands_target.mean(dim=0)
	H = center_lands_source.transpose(0,1).matmul(center_lands_target)
	u, e, v = torch.svd(H)
	R = v.matmul(u.transpose(0,1)).detach()

	# check rotation is not a reflection
	if R.det() < 0.0:
		v[:, -1] *= -1
		R = v.matmul(u.transpose(0,1)).detach()
		
	######################################################################################################
	
	
	
	
		
	t_x = torch.tan(half_angle/2.0 + torch.pi/4) #### calculated in my circle-inversion diagram
	
	new_W = 1.0/(t_x - 1.0)
	new_E = 1.0/(t_x + 1.0)
	
	new_c = 0.5*(new_W + new_E)
	new_r = new_W - new_c
	
	return (t_x, new_c, new_r, R)
	






def invert_sphere1(P, inversion_sig): #### take two points to the north and south pole

	t_x, new_c, new_r, R = inversion_sig
	
	P_rot = P @ R.T #align two landpoints with v=pi

	P_trans = P_rot + torch.tensor([t_x, 0.0, 0.0], device = P.device) #translate
	inverted_P = ((1.0/torch.sum(P_trans*P_trans, axis=1))*P_trans.T).T
	centred_P = inverted_P - torch.tensor([new_c, 0.0, 0.0], device = P.device)

	scaled_P =  centred_P / new_r 
	
	return scaled_P
	
	
	
	
def invert_sphere2(P, inversion_sig): #### reverse the inversion that took two points to the north and south pole.
	
	t_x, new_c, new_r, R = inversion_sig
	
	scaled_P =  P * new_r # scale the sphere
	uncentred_P = scaled_P + torch.tensor([new_c, 0.0, 0.0], device = P.device)
	inverted_P = ((1.0/torch.sum(uncentred_P**2, axis=1))*uncentred_P.T).T
	P_untrans = inverted_P - torch.tensor([t_x, 0.0, 0.0], device = P.device)
	
	P_unrotated = P_untrans @ R
	
	return P_unrotated
	
	

def compute_half_mobius_sig(S_corresp, special_indices = [0,1,2]):
	a,b,c = special_indices
	
	### Compute the invesion signature for taking points A and B to the North and South pole. ###
	### A,B,C |--> A', B', C' ### 
	sig1 = compute_inversion_sig(S_corresp, special_indices = [a,b])
	### Compute images, and rotate: A'=N, B'=S, C' |--> A''=W, B''=E, C'' ### 
	S_corresp2 = invert_sphere1( S_corresp , sig1 )
	
	R1 = torch.tensor([[0.0, 0.0, 1.0],
								[0.0, 1.0, 0.0],
								[-1.0,  0.0, 0.0]])
								
	S_corresp3 = S_corresp2 @ R1 ### a,b have moved to (-1,0,0) and (1,0,0) resp.


	alpha = torch.atan2(S_corresp3[c,2], S_corresp3[c,1])
	

			
	R2 = (torch.tensor([[1.0, 0.0, 0.0],
			[0.0, torch.cos(alpha), -torch.sin(alpha)],
			[0.0, torch.sin(alpha), torch.cos(alpha)]])
			@ torch.tensor([[1.0, 0.0, 0.0],
							[0.0, 0.0, 1.0],
							[0.0, -1.0, 0.0]]))
	
	S_corresp4 = S_corresp3 @ R2 ### a and b are still at (-1,0,0) and (1,0,0) and c is aligned to xz plane
	

	### Compute the inversion signature for taking point C'' to North pole ###
	
	
	inversion_angle = torch.acos(torch.sum(S_corresp2[a,:]*S_corresp2[c,:])) # find half angle for inversion - ie angle between a and c.
		
	######################################################################################################
		
	t_x = torch.tan(inversion_angle/2.0 + torch.pi/4) #### calculated in my circle-inversion diagram
	
	new_W = 1.0/(t_x - 1.0)
	new_E = 1.0/(t_x + 1.0)
	
	new_c = 0.5*(new_W + new_E)
	new_r = new_W - new_c
	
	sig2 = (t_x, new_c, new_r, (R1@R2).T)
	
	return(sig1, sig2)


def compute_full_mobius_sig(S_corresp1, S_corresp2, special_indices1 = [0,1,2], special_indices2=[0,1,2]):
	sig1,sig2 = compute_half_mobius_sig(S_corresp1, special_indices = special_indices1)
	sig3,sig4 = compute_half_mobius_sig(S_corresp2, special_indices = special_indices2)
	
	return (sig1,sig2,sig3,sig4)







def full_mobius_transform(points, mobius_sig):
	sig1,sig2,sig3,sig4 = mobius_sig
	
	res1 = invert_sphere1(points, sig1)
	res2 = invert_sphere1(res1, sig2)
	res3 = invert_sphere2(res2, sig4)
	res4 = invert_sphere2(res3, sig3)
	
	return res4








	

def inversions_test1():

	corresp = torch.randn((2,3))
	corresp = (corresp.T/torch.sqrt(torch.sum(corresp**2, axis = 1))).T

	poles = torch.tensor([[0.0, 0.0, -1.0],[0.0, 0.0, 1.0]])

	sig = compute_inversion_sig(corresp)
	
	result1 = invert_sphere1(corresp, sig)
	print('forwards error:',torch.sum((result1 - poles )**2))

	result2 = invert_sphere2(poles, sig)
	print('backwards error:',torch.sum((result2 - corresp )**2))

def inversions_test2():
	tm = trimesh.load('/Users/romywilliamson/Documents/SphericalNS/spherical2/data/analytic/sphere/sphere3.obj')
	
	P = torch.tensor(tm.vertices, dtype=torch.float32)
	F = tm.faces
	
	corresp = torch.randn((2,3))
	corresp = (corresp.T/torch.sqrt(torch.sum(corresp**2, axis = 1))).T

	poles = torch.tensor([[0.0, 0.0, -1.0],[0.0, 0.0, 1.0]])

	sig = compute_inversion_sig(corresp)

	P_dash1 = invert_sphere1(P, sig)
	

	P_dash2 = invert_sphere2(P_dash1, sig)
	
	igl.write_triangle_mesh('/Users/romywilliamson/Desktop/SaturnVDesktop/p.obj', P.numpy(), F)
	igl.write_triangle_mesh('/Users/romywilliamson/Desktop/SaturnVDesktop/pdash1.obj', P_dash1.numpy(), F)
	igl.write_triangle_mesh('/Users/romywilliamson/Desktop/SaturnVDesktop/pdash2.obj', P_dash2.numpy(), F)
	
	
	import matplotlib.pyplot as plt
	
	plt.plot(  (torch.sum(P_dash1**2, axis=1))  )
	plt.show()
	
	plt.plot(  (torch.sum(P_dash2**2, axis=1))  )
	plt.show()
	
	plt.plot(  (torch.sum((P - P_dash2)**2, axis=1))  )
	plt.show()


def inversions_test3(): ### this test is best.
    ### choose 3 source landmarks
	corresp1 = torch.randn((2,3))
	corresp1 = (corresp1.T/torch.sqrt(torch.sum(corresp1**2, axis = 1))).T

	poles = torch.tensor([[0.0, 0.0, 1.0],[0.0, 0.0, -1.0]])
    
    ### find inversion sig for mapping those landmarks to the poles.
	sig1 = compute_inversion_sig(corresp1)

	result1 = invert_sphere1(corresp1, sig1)
	print('forwards error:',torch.sum((result1 - poles )**2))
	
	
	### choose 3 target landmarks
	corresp2 = torch.randn((2,3))
	corresp2 = (corresp2.T/torch.sqrt(torch.sum(corresp2**2, axis = 1))).T
	
	### compute inversion sig for mapping the poles to the target landmarks
	sig2 = compute_inversion_sig(corresp2)

	result2 = invert_sphere2(result1, sig2)
	print('backwards error:',torch.sum((result2 - corresp2 )**2))


	
def mobius_test1(): ### check half mobius calculation manually
	corresp = torch.randn((3,3))
	corresp = (corresp.T/torch.sqrt(torch.sum(corresp**2, axis = 1))).T
	
	mobius_sig = compute_half_mobius_sig(corresp)
	
	res1 = invert_sphere1(corresp, mobius_sig[0])

	res2 = invert_sphere1(res1 , mobius_sig[1])
	
	print(res2.round(decimals=2)) ### should be [[1,0,0],[-1,0,0],[0,0,1]]
	
	res3 = invert_sphere2(res2, mobius_sig[1])
	
	res4 = invert_sphere2(res3, mobius_sig[0])
	print((res4 - corresp).round(decimals=2))
	return 

def mobius_test2(): ### check half mobius calculation manually
	corresp1 = torch.randn((3,3))
	corresp1 = (corresp1.T/torch.sqrt(torch.sum(corresp1**2, axis = 1))).T
	
	corresp2 = torch.randn((3,3))
	corresp2 = (corresp2.T/torch.sqrt(torch.sum(corresp2**2, axis = 1))).T
	
	sig1,sig2,sig3,sig4 = compute_full_mobius_sig(corresp1, corresp2)
	
	res1 = invert_sphere1(corresp1, sig1)
	res2 = invert_sphere1(res1, sig2)
	res3 = invert_sphere2(res2, sig4)
	res4 = invert_sphere2(res3, sig3)
	
	print((res4 - corresp2).round(decimals=2))
	

def mobius_test3(): ### check half mobius calculation manually
	corresp1 = torch.randn((3,3))
	corresp1 = (corresp1.T/torch.sqrt(torch.sum(corresp1**2, axis = 1))).T
	
	corresp2 = torch.randn((3,3))
	corresp2 = (corresp2.T/torch.sqrt(torch.sum(corresp2**2, axis = 1))).T
	
	mobius_sig = compute_full_mobius_sig(corresp1, corresp2)
	
	result = full_mobius_transform(corresp1, mobius_sig)
	
	print((result - corresp2).round(decimals=2))
	
	
	return

	
def mobius_test4():
	tm = trimesh.load('/Users/romywilliamson/Documents/SphericalNS/spherical6/data/analytic/sphere/sphere4.obj')
	
	P = torch.tensor(tm.vertices, dtype=torch.float32)
	F = tm.faces
	
	for i in range(5):
		corresp1 = torch.randn((3,3))
		corresp1 = (corresp1.T/torch.sqrt(torch.sum(corresp1**2, axis = 1))).T
		
		corresp2 = torch.randn((3,3))
		corresp2 = (corresp2.T/torch.sqrt(torch.sum(corresp2**2, axis = 1))).T
		
		

		mobius_sig = compute_full_mobius_sig(corresp1, corresp2)
		P_dash = full_mobius_transform(P, mobius_sig)
			
		igl.write_triangle_mesh('../data/distortion_expmt/p.obj', P.numpy(), F)
		igl.write_triangle_mesh('../data/distortion_expmt/pdash'+str(i)+'.obj', P_dash.numpy(), F)
	
		



def make_variations(spherepath):
	tm = trimesh.load(spherepath)
	
	P = torch.tensor(tm.vertices, dtype=torch.float32)
	F = tm.faces
	
	corresp1 = torch.Tensor([[0.0, 0.0, 1.0],
						[0.0, 1.0, 0.0],
						[0.0, 0.0, -1.0]])
				
	for i in [1,2,3]:
		#corresp1 = torch.randn((3,3))
		#corresp1 = (corresp1.T/torch.sqrt(torch.sum(corresp1**2, axis = 1))).T
		
		#corresp2 = torch.randn((3,3))
		#corresp2 = (corresp2.T/torch.sqrt(torch.sum(corresp2**2, axis = 1))).T

		theta = (i+1) * (torch.pi / 2) / 4 
		corresp2 = torch.Tensor([[0.0,np.sin(theta), np.cos(theta)],
							[0.0, 1.0, 0.0],
							[0.0, np.sin(theta), - np.cos(theta)]])
		
		
		

		mobius_sig = compute_full_mobius_sig(corresp1, corresp2)
		P_dash = full_mobius_transform(P, mobius_sig)
			
		
		igl.write_triangle_mesh('../data/distortion_expmt/var'+str(i)+'.obj', P_dash.numpy(), F)

	
	for i in [4,5,6]:
		P_dash = P.clone()
		#P_dash[:,1] = ((i+1)*P_dash[:,1])**3
		P_dash[:,1] = 2*(i+1)*P_dash[:,1]


		P_dash = P_dash / torch.sqrt(P_dash.pow(2).sum(dim=1, keepdim=True))

		igl.write_triangle_mesh('../data/distortion_expmt/var'+str(i)+'.obj', P_dash.numpy(), F)



	igl.write_triangle_mesh('../data/distortion_expmt/original.obj', P.numpy(), F)


	
#make_variations()

#mobius_test4()

#inversions_test3()

