import torch
from torch import autograd as Grad
from torch.nn import functional as F
from torch.nn import Module
import numpy as np
import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

if torch.cuda.is_available():
	try:
		from torch_batch_svd import svd
	except:
		from torch import svd
else:
	from torch import svd


class DifferentialModule(Module):

	# ================================================================== #
	# =================== Compute the gradient ========================= #
	def gradient(self, out, wrt, allow_unused=False):

		B = 1 if len(out.size()) < 3 else out.size(0)
		N = out.size(0) if len(out.size()) < 3 else out.size(1)
		R = out.size(-1)
		C = wrt.size(-1)

		gradients = []
		for dim in range(R):
			out_p = out[..., dim].flatten()

			select = torch.ones(out_p.size(), dtype=torch.float32).to(out.device)

			gradient = Grad.grad(outputs=out_p, inputs=wrt, grad_outputs=select, create_graph=True, allow_unused=allow_unused)[0]
			gradients.append(gradient)

		if len(out.size()) < 3:
			J_f_uv = torch.cat(gradients, dim=1).view(N, R, C)
		else:
			J_f_uv = torch.cat(gradients, dim=2).view(B, N, R, C)

		return J_f_uv

	def backprop(self, out, wrt):

		select = torch.ones(out.size(), dtype=torch.float32).to(out.device)

		J = Grad.grad(outputs=out, inputs=wrt, grad_outputs=select, create_graph=True)[0]
		J = J.view(wrt.size())
		return J

	
	def parametrise(self, wrt):
		'''Compute an orthonormal basis (with consistent orientation) for the tangent planes on the sphere '''
		  
		
		X,Y,Z = wrt[:,0],wrt[:,1],wrt[:,2]
		### Relabel x,y,z coordinates that are a certain distance from the equator,
		### so that we never need to deal with N/S pole singularities

		mask = abs(Z)<0.5		
		x,y,z = X*mask + Y*(~mask), Y*mask + Z*(~mask), Z*mask + X*(~mask)
		
		# sphere parametrisation is ( sin u * cos v,   sin u * sin v,    cos u ) --- u in 0-2pi, v in 0-pi
		
		#compute u,v trig quantites in terms of x,y,z
		cos_u = z
		sin_u = torch.sqrt(x**2 + y**2)
		sin_v = y/sin_u
		cos_v = x/sin_u
		
		###### (dg_du, dg_dv) is the orthonormal R matrix, in the notation of the SNS paper
		###### [ The notation dg_du, dg_dv makes sense when we imagine that 'g' is the rigid transform
		######  that takes a section of the 2d plane around the origin to the location of the tangent plane on the sphere
		###### at the point with spherical polar coordinates u,v . ]
		dg_du = torch.stack((cos_u*cos_v,   cos_u*sin_v,  -1.0*sin_u ) ).transpose(0,1)
		dg_dv = torch.stack((-1.0*sin_v, cos_v,  0.0*cos_v  )).transpose(0,1)
		
		###### This is the jacobian if 'g' is just the spherical polar coordinates map (not an isometry).
		#dg_du = torch.stack((cos_u*cos_v,   cos_u*sin_v,  -1.0*sin_u ) ).transpose(0,1)  
		#dg_dv = torch.stack((-1.0*sin_u*sin_v,   sin_u*cos_v,  0.0*sin_u  )).transpose(0,1)
		
		dg_du = dg_du * mask.unsqueeze(-1) + dg_du[:,(2,0,1)]*(~mask.unsqueeze(-1)) ### undo the coordinate relabelling
		dg_dv = dg_dv * mask.unsqueeze(-1) + dg_dv[:,(2,0,1)]*(~mask.unsqueeze(-1))
			
		return dg_du, dg_dv
	
				
	def compute_normals(self, jacobian=None, out=None, wrt=None, return_grad=False):

		if jacobian is None:
			jacobian = self.gradient(out=out, wrt=wrt) #### 3x3 matrix
		jacobian3x3 = jacobian
		
	
		dg_du, dg_dv = self.parametrise(wrt)
		
		dx_du = (jacobian3x3 @ dg_du.unsqueeze(-1)).squeeze(-1)

		dx_dv = (jacobian3x3 @ dg_dv.unsqueeze(-1)).squeeze(-1)
		
		cross_prod = torch.cross(dx_du, dx_dv, dim=-1)

		##### normalize, except when the cross products are very small
		idx_small = cross_prod.pow(2).sum(-1) < 10.0**-7
		normals = F.normalize(cross_prod, p=2, dim=-1)  # (..., N, 3)
		normals[idx_small] = cross_prod[idx_small]
		
		jacobian3x2 = torch.stack((dx_du, dx_dv)) #also known as geo_jacobian
		jacobian3x2 = jacobian3x2.permute(1,2,0)
		
	
		if return_grad:
			return normals, jacobian3x2, jacobian3x3, dg_du, dg_dv
		
		return normals
		
	
	# ================================================================== #
	# ================ Compute first fundamental form ================== #
	def compute_FFF(self, geo_jacobian=None, out=None, wrt=None, return_grad=False): 
			
		if geo_jacobian is None:
			normals,geo_jacobian, _, dg_da, dg_db = self.compute_normals(jacobian=None, out=out, wrt=wrt, return_grad=True)

		FFF = geo_jacobian.transpose(1,2) @ geo_jacobian #FFF as a batched matrix

		# Extracts E, F, G terms
		I_E = FFF[:,0,0]
		I_F = FFF[:,0,1]
		I_G = FFF[:, 1,1]

		if return_grad:
			return I_E, I_F, I_G, geo_jacobian, normals
		return I_E, I_F, I_G

	# ================================================================== #
	# ================ Compute second fundamental form ================= #
	def compute_SFF(self, jacobian=None, out=None, wrt=None, return_grad=False, return_normals=False):
		normals, geo_jacobian, _, dg_da, dg_db = self.compute_normals(jacobian=None, out=out, wrt=wrt, return_grad=True)
		
		# a,b are local coordinates on the sphere
		# r is the position on the surface
		# ra, rb represent dr_da, dr_db, and similarly raa, rab, rbb are the second derivatives.
		
		ra = geo_jacobian[..., 0]
		rb = geo_jacobian[..., 1]

		#find derivatives of ra, rb wrt sphere positions
		ra_deriv = self.gradient(out = ra, wrt = wrt)
		rb_deriv = self.gradient(out = rb, wrt = wrt)
		
		#do the chain rule to find the derivatives wrt the R^2 positions
		raa = (ra_deriv @ (dg_da.unsqueeze(1).transpose(1,2)) ).squeeze()
		rab = (ra_deriv @ (dg_db.unsqueeze(1).transpose(1,2)) ).squeeze()
		rbb = (rb_deriv @ (dg_db.unsqueeze(1).transpose(1,2)) ).squeeze()
		
		#compute second fundamental form coefficients
		L = (raa * normals).sum(-1)
		M = (rab * normals).sum(-1)
		N = (rbb * normals).sum(-1)

		if return_grad and return_normals:
			return L, M, N, jacobian, normals
		if return_grad:
			return L, M, N, jacobian
		return L, M, N
		
		
	def compute_area_distortion(self, out=None, wrt=None):
		''' compute local area distortion from the sphere to the surface, using FFF '''
		E,F,G = self.compute_FFF(out = out, wrt = wrt)
		distortion = torch.sqrt(E*G - F.pow(2))	
		return distortion

	# ================================================================== #
	# ================ Compute curvature from FFF, SFF ================== #
	def compute_curvature(self, jacobian=None, out=None, wrt=None, compute_principal_directions=False, prevent_nans=True):
		I_E, I_F, I_G, geo_jacobian, normals = self.compute_FFF(geo_jacobian=None, out=out,wrt=wrt, return_grad=True)
		L, M, N = self.compute_SFF(jacobian=jacobian, out=out, wrt=wrt, return_grad=False, return_normals=False)
		
		
		#prevent Nans in SFF
		if prevent_nans:
			big = 10000000000*torch.ones_like(L, dtype=L.dtype)
			
			L = L * (torch.abs(L)<big)
			M = M * (torch.abs(M)<big)
			N = N * (torch.abs(L)<big)
		
		####### principal curvatures are the eigenvalues of (FFFinv)(SFF)
		####### if the directions are not needed, we do not compute them.
		####### mean curv is arithmetic mean of principal curvatures.
		####### Gauss curv is product of principal curvatures.
	
		####### coefficients of the quadratic equation det (SFF - kFFF) = 0
		####### i.e. this is the characteristic equation
		A = (I_E * I_G - I_F.pow(2))#FFFdet
		B = 2 * M * I_F - (I_E * N + I_G * L)
		C = (L * N - M.pow(2))#SFFdet
		
		H = -B/(2.0*A)#mean curvature is sum of eigenvalues, (sum of roots) i.e. H = (LG-2MF+NE)/(2*(EG-F^2))
		K = C / A #Gauss curvature is product of eigenvalues, (product of roots) i.e. K = (LN-M^2)/(EG-F^2)
		H *= -1 #flip sign of H to fit with graphics convention
		
		if compute_principal_directions:
			#### principal curvatures and principal curvature directions ####
			k1 = (-B - torch.sqrt(B**2 - 4*A*C))/(2.0*A)#always smaller principal curvature
			k2 = (-B + torch.sqrt(B**2 - 4*A*C))/(2.0*A)#always bigger principal curvature
			
			e1 = geo_jacobian[:,:,0]
			e2 = geo_jacobian[:,:,1]
			
			x1 = (k1*I_E - L)/(M - k1*I_F)  ###need to adjust for when denominator is zero.
			
			if prevent_nans:
				big = 10000000*torch.ones_like(x1)
				x1 = torch.where(torch.abs(x1)<big, x1, 0.0 )
			
			dir1 = torch.stack((torch.ones_like(x1), x1)).T
			dir1 = (dir1[:,0].T*e1.T + dir1[:,1].T*e2.T).T
			dir1 = F.normalize(dir1, p=2, dim=1)

			x2 = (k2*I_E - L)/(M - k2*I_F)
			
			if prevent_nans:
				big = 10000000*torch.ones_like(x1)
				x2 = torch.where(torch.abs(x2)<big, x2, 0.0 )
			
			dir2 = torch.stack((torch.ones_like(x2), x2)).T
			dir2 = (dir2[:,0].T*e1.T + dir2[:,1].T*e2.T).T
			dir2 = F.normalize(dir2, p=2, dim=1)
			
			
			#for testing, we can compute normals as cross prod of principal curvature directions
			#cross = torch.cross(e1,e2, dim=-1) 
			#normals = F.normalize(cross, p=2, dim=1)
			
			## fix signs
			frame = (torch.stack([dir1, dir2, normals])).transpose(0,1)
			signs = torch.linalg.det(frame)
			dir1 = (dir1.T*signs.T).T
			
			# compute other curvatures derived from principal curvatures
			abs_k1 = abs(k1)
			abs_k2 = abs(k2)
			MAC = abs_k1 * (abs_k1>=abs_k2) + abs_k2*(abs_k1<abs_k2) #maximum absolute curvature
			SMAC = k1 * (abs_k1>=abs_k2) + k2 * (abs_k1<abs_k2) #signed maximum absolute curvature
			
			return H,K, MAC,SMAC, (dir1,dir2), (k1, k2), normals
											
		return H, K, None,None,None,None,None
			
			
	def laplace_beltrami_divgrad(self, out=None, wrt=None, f=None, f_defined_on_sphere=False):
		''' computes LBO on a function f defined onn the surface, as the surface divergence of the surface gradient of f '''
		
		normals, _, jacobian3D, _,_ = self.compute_normals(out=out, wrt=wrt, return_grad=True)
		
		inv_jacobian3D = torch.linalg.inv(jacobian3D)
		 
		### DERIVATIVE 1
		if not f_defined_on_sphere:
			df = self.gradient(out = f.unsqueeze(-1), wrt=out).squeeze() ## usual euclidean grad formula
		else:
			#when f is defined on the sphere not the surface, computing the grad requires chain rule with inv_jacobian3D
			df = (self.gradient(out = f.unsqueeze(-1), wrt=wrt).squeeze().unsqueeze(1) @ inv_jacobian3D).squeeze()

		### SURFACE GRADIENT (F)
		## the Riemannian grad of f on the surface (nx3) is the euclidean grad df, minus the normal component of df
		F = df - torch.sum(df*normals, axis=1).unsqueeze(-1)*normals
		
		
		### DERIVATIVE 2 
		dF = self.gradient(out = F, wrt=wrt) @ inv_jacobian3D  ##differentiate the surface grad
		
		#euclidean divergence is dF1_dx1 + dF2_dx2 + dF3_dx3
		divF = dF[:,0,0] + dF[:,1,1] + dF[:,2,2]
		
		# to find the 'surface divergence' we need to remove the component of divF
		# that is due to variation of F in the normal direction
		normals_term = ( normals.unsqueeze(1) @ dF @ normals.unsqueeze(2) ).squeeze()	
		LB_f = divF - normals_term
		
		print('laplace beltrami divgrad finished. Computed surface divergence of surface gradient.')
		
		return LB_f
		

	def laplace_beltrami_MC(self, normals, meancurv, f, grad_f=None, hessian_f=None): 
	
		'''This version is not actually differential because we compute the gradient and hessian
		before passing to this function, and the normals and mean curvature are differentially computed but 
		not in this function.
		It computes the LB operator on a function using the formula
		LB(f) = Delta (f) - 2H grad(f).n - nT Hess(f) n
		We can check this formula by applying it to spherical harmonics in Cartesian form
		(see the table in https://cs.dartmouth.edu/~wjarosz/publications/dissertation/appendixB.pdf).
		'''
		
		divgrad = hessian_f[:,0,0] + hessian_f[:,1,1] + hessian_f[:,2,2]
		
		meancurv_term = - 2 * meancurv * ((grad_f*normals).sum(-1))
		
		hessian_term = -1*( normals.unsqueeze(1) @ hessian_f @ normals.unsqueeze(2) ).squeeze()
		
		LB_f =   divgrad + meancurv_term + hessian_term # (euclidean laplacian) - ( 2 Hn .grad(f) ) -  ( nT Hessian n )
		
		print('computed LB with meancurv formula')
		
		return LB_f
			
