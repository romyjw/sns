import torch
import numpy as np


def stereographic(points_R3):#sphere to complex plane. Checked.
	#input: n x 3 real array. Output: 1D complex numbers array.
	
	#print('stereographic shape', points_R3.shape)
	x,y,z = (points_R3[:,0], points_R3[:,1], points_R3[:,2])
	if points_R3.isnan().any():
            raise ValueError('input to stereographic is bad')
            
	
	points_Cstar = (x + y*(0.0+1.0j))/(1.0 - z) ##### it's regularised
	if points_Cstar.isnan().any():
            raise ValueError('output of stereographic is bad')
	return points_Cstar

def stereographic_inv(points_Cstar):#complex plane to sphere. Checked.

	x,y = (points_Cstar.real, points_Cstar.imag)	
	
	try:
	    points_R3 = (torch.stack((2*x, 2*y, x**2 + y**2 - 1.0))/(x**2 + y**2 + 1.0)).T
	except:
	    points_R3 = (np.stack((2*x, 2*y, x**2 + y**2 - 1.0))/(x**2 + y**2 + 1.0)).T
	return points_R3
	

def stereographic2(points_R3):#sphere to complex plane. Checked.
	#input: n x 3 real array. Output: 2D real numbers array.
	
	#print('stereographic shape', points_R3.shape)
	
	z = points_R3[:,2]
	xy = points_R3[:,:2]
	
	
	if points_R3.isnan().any():
            raise ValueError('input to stereographic is bad')
           

	#points_Cstar = ((xy + 0.0001*(xy-0.1)**2).T/(0.9999 - z)).T  ###### various attempts at regularisation
	points_Cstar = ((xy.T/(1.0 - z))).T

	if points_Cstar.isnan().any():
            raise ValueError('output of stereographic is bad')
	return points_Cstar

def stereographic_inv2(points_Cstar):#complex plane to sphere. Checked.
	##### Oh. There is a simpler formula:
	##### (2r, 2s, r**2 + s**2 - 1 )/(r**2 + s**2 + 1). Easier for finding derivative.

	x,y = (points_Cstar[:,0], points_Cstar[:,1])	
	
	try:
	    points_R3 = (torch.stack((2*x, 2*y, x**2 + y**2 - 1.0))/(x**2 + y**2 + 1.0)).T
	except:
	    points_R3 = (np.stack((2*x, 2*y, x**2 + y**2 - 1.0))/(x**2 + y**2 + 1.0)).T
	return points_R3

def stereographic_inv_old(points_Cstar):#complex plane to sphere. Checked.
	##### Oh. There is a simpler formula:
	##### (2r, 2s, r**2 + s**2 - 1 )/(r**2 + s**2 + 1). Easier for finding derivative.

	x,y = (points_Cstar.real, points_Cstar.imag)	
	a = 1 + x**2 + y**2
	b = -2*(x**2 + y**2)
	c = x**2 + y**2 - 1
	
	d = b**2 - 4*a*c
	sol1 = (-1.0*b + (d)**0.5)/(2.0*a)
	sol2 = (-1.0*b - (d)**0.5)/(2.0*a)
	
	T = sol1 * (sol1<sol2) + sol2 * (sol2<sol1)
	try:
	    points_R3 = torch.stack(((1-T)*x, (1-T)*y, T)).T
	except:
	    points_R3 = np.stack(((1-T)*x, (1-T)*y, T)).T
	return points_R3
	
	

def triplet_defined_mobius0(C_correspondences, Z, invert=False):#mobius in the complex plane
	### input: tuple of six complex numbers, and a 1D array of complex numbers
	### first three get sent to 0,1, infinity. Testing purposes.

	w1,w2,w3 = C_correspondences
	
	######### NO MENTION OF z1,z2,z3

	return ((Z - w1)/(Z-w3))*((w2-w3)/(w2-w1))

	
	
def triplet_defined_sphere_mobius0(S_correspondences, P, invert=False):#mobius on the sphere
	### first three get sent to S,(1,0,0), N . Testing purposes.

	C_correspondences = stereographic(S_correspondences)
	
	C_points_in = stereographic(P)
	
	C_points_out = triplet_defined_mobius0(C_correspondences, C_points_in)
	
	return stereographic_inv(C_points_out)	
	
	

def triplet_defined_mobius1(C_correspondences, Z):#mobius in the complex plane. Checked.
	### input: tuple of six complex numbers, and a 1D array of complex numbers
	###### composition of mobius maps: (w1,w2,w3)--> (0,1, inf)--> (z1,z2,z3)

	w1,w2,w3,z1,z2,z3 = C_correspondences
	I = np.ones_like(Z)
	
	gamma_w = (w2-w3)/(w2-w1)
	gamma_z = (z2-z3)/(z2-z1)
	
	A = gamma_z*z1 - z3*gamma_w
	B = z3*gamma_w*w1 - w3*gamma_z*z1
	C = gamma_z - gamma_w
	D = gamma_w*w1 - w3*gamma_z
	
	
	#print(' did triplet mobius 1 work?   ', z1,     (A*w1 + B)/(C*w1 + D)     )
	#print(' did triplet mobius 1 work?   ', z2,     (A*w2 + B)/(C*w2 + D)     )
	#print(' did triplet mobius 1 work?   ', z3,     (A*w3 + B)/(C*w3 + D)     )
	
	return (A*Z + B*I)/(C*Z + D*I)









	

def mobius(mobius_sig,Z, lib='torch'):#mobius in the complex plane. Checked.
	####### complex plane to complex plane with complex A,B,C,D ###########
	A,B,C,D = mobius_sig
	if lib=='torch':
		I = torch.ones_like(Z)
	else:
		I = np.ones_like(Z)

	result = (A*Z + B*I)/(C*Z + D*I)
	if result.isnan().any():
	    print('exiting because nan occurred in mobius')
	    exit()
	return result
	
def sphere_mobius(mobius_sig, P, lib='torch'):
	####### sphere to sphere with (complex) A,B,C,D ###########
	A,B,C,D = mobius_sig
	
	C_points_in = stereographic(P)	
	C_points_out = mobius(mobius_sig, C_points_in, lib=lib)
	
	return stereographic_inv(C_points_out)


def find_mobius_from_3corresp(S_correspondences):
	C_correspondences = stereographic(S_correspondences)
	#find the mobius transformations that map the first 3 source landmarks to S, x=1, N
        # and then maps N, S, x=1 to the first 3 target landmarks.
	w1,w2,w3,z1,z2,z3 = C_correspondences
	mobius_sig = []
	############# ((Z - w1)/(Z-w3))*((w2-w3)/(w2-w1))

	scale = (w2 - w3)/(w2 - w1)
	A = scale
	B = -1.0*scale*w1
	C = 1.0 + 0.0j
	D = -1.0*w3
	
	mobius_sig.append((A,B,C,D))
	#mobius_sig.append((1.0,0.0,0.0,1.0))
	############# ((Z - z1)/(Z-z3))*((z2-z3)/(z2-z1))
	
	scale2 = (z2 - z3)/(z2 - z1)
	A2 = scale2
	B2 = -1.0*scale2*z1
	C2 = 1.0 + 0.0j
	D2 = -1.0*z3
	
	mobius_sig.append((D,-1.0*B,-1.0*C,A))

	#mobius_sig.append((D2,-1.0*B2,-1.0*C2,A2))
	#mobius_sig.append((1.0,0.0,0.0,1.0))
	#mobius_sig.append((A2,B2,C2,D2))
	return mobius_sig ####list of two complex 4-tuples. for mob transform to poles and back.



	
	
def triplet_defined_sphere_mobius1(S_correspondences, P):#mobius on the sphere. Checked.
	#input: tuple of 6 sphere points, P = array of all sphere points
	print('S corresp',S_correspondences.shape)
	
	C_correspondences = stereographic(S_correspondences)
	
	C_points_in = stereographic(P)
	
	C_points_out = triplet_defined_mobius1(C_correspondences, C_points_in)
	
	return stereographic_inv(C_points_out)	
	

	

def triplet_defined_mobius(C_correspondences, Z):#mobius in the complex plane
	### input: tuple of six complex numbers, and a 1D array of complex numbers

	w1,w2,w3,z1,z2,z3 = C_correspondences
	
	A = (w2 - w3)/(w2 - w1)
	B = (z2 - z3)/(z2 - z1)
	
	C = (Z - z1)/(Z - z3)
	
	return (A*w1 - B*C*w3)/(A - B*C)
	


	
	

def triplet_defined_sphere_mobius(S_correspondences, P):#mobius on the sphere


	C_correspondences = stereographic(S_correspondences)
	
	C_points_in = stereographic(P)
	
	C_points_out = triplet_defined_mobius(C_correspondences, C_points_in)
	
	return stereographic_inv(C_points_out)
	
	
	
