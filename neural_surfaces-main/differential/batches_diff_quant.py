import numpy as np
import torch
from math import ceil
from torch.nn import functional as F

class batches_diff_quant():
	def __init__(self, vertices, model, diffmod, batch_size, scaling=1.0):
		n = vertices.shape[0]
		
		self.vertices = vertices
		self.model = model
		self.diffmod = diffmod
		self.batch_size = batch_size
		
		self.num_batches = ceil(n/batch_size )
		self.all_output_vertices = np.zeros_like(vertices)
		self.all_normals = np.zeros_like(vertices)
		self.all_H = np.zeros(n)
		self.all_K = np.zeros(n)
		self.all_directions = [np.zeros((n, 3)), np.zeros((n, 3))]## two principal curvature directions
		self.all_distortions = np.zeros(n)
		self.all_principals = [np.zeros(n),np.zeros(n)]
		self.all_beltrami_H = np.zeros(n)
		self.all_area_distortions = np.zeros(n)
		self.all_beltrami_on_X = np.zeros((n,3))
		self.all_beltrami_on_scalar = np.zeros(n)
		self.n = n
		
		self.scaling = scaling
		
	def get_indexed_batch(self, i, requires_grad=True):
		print('Computing batch ', i, ' out of ', self.num_batches,'(',self.n,' vertices total)' )
		
		
			
		start = self.batch_size*i 
		end = min(self.batch_size*(i+1), self.n) 
		
		tensorvertices = torch.Tensor(self.vertices[start : end , :])
		
		if requires_grad==True:
			tensorvertices.requires_grad=True
	
		output_vertices = self.scaling * self.model.forward(tensorvertices)
	
		self.all_output_vertices[start: end , :] = output_vertices.detach().numpy().copy()
		
		return tensorvertices, output_vertices, start, end
	
	def compute_SNS(self):
		for i in range(self.num_batches):
			tensorvertices, output_vertices, start, end = self.get_indexed_batch(i)
	
		
	def compute_normals(self):
		for i in range(self.num_batches):
			tensorvertices, output_vertices, start, end = self.get_indexed_batch(i)
			
			normals = self.diffmod.compute_normals(out = output_vertices, wrt = tensorvertices)
			self.all_normals[start : end , :] = normals.detach().numpy().copy()
	
	def compute_area_distortions(self):
		for i in range(self.num_batches):
			tensorvertices, output_vertices, start, end = self.get_indexed_batch(i)
			
			area_distortion = self.diffmod.compute_area_distortion(out=output_vertices, wrt=tensorvertices)
			
			self.all_area_distortions[start : end ] = area_distortion.detach().numpy().copy()
	
	def compute_curvature(self):
		for i in range(self.num_batches):
			tensorvertices, output_vertices, start, end = self.get_indexed_batch(i)
			
			H,K,_,_,_,_,_ = self.diffmod.compute_curvature(out = output_vertices, wrt = tensorvertices, compute_principal_directions=False, prevent_nans=False)

			self.all_H[start : end] = H.detach().numpy().copy()
			self.all_K[start : end] = K.detach().numpy().copy()
	
	def compute_directions(self):
		
		for i in range(self.num_batches):
			tensorvertices, output_vertices, start, end = self.get_indexed_batch(i)
			
			H,K, MAC,SMAC, (dir1,dir2), (k1, k2), normals = self.diffmod.compute_curvature(out = output_vertices, wrt = tensorvertices, compute_principal_directions=True, prevent_nans=False)

			self.all_H[start : end] = H.detach().numpy().copy()
			self.all_K[start : end] = K.detach().numpy().copy()
			
			self.all_normals[start : end , :] = normals.detach().numpy().copy()
			
			self.all_directions[0][start : end] = dir1.detach().numpy().copy()
			self.all_directions[1][start : end] = dir2.detach().numpy().copy()
		
		
	
	def compute_beltrami_on_X(self):
		for i in range(self.num_batches):
			tensorvertices, output_vertices, start, end = self.get_indexed_batch(i)
			
			beltrami_result_x = self.diffmod.laplace_beltrami_divgrad( out=output_vertices, wrt=tensorvertices, f=output_vertices[:,0])
			beltrami_result_y = self.diffmod.laplace_beltrami_divgrad( out=output_vertices, wrt=tensorvertices, f=output_vertices[:,1])
			beltrami_result_z = self.diffmod.laplace_beltrami_divgrad( out=output_vertices, wrt=tensorvertices, f=output_vertices[:,2])
			beltrami_result = torch.stack((beltrami_result_x, beltrami_result_y, beltrami_result_z)).transpose(0,1)
			
			self.all_beltrami_on_X[start:end] = beltrami_result.detach().numpy().copy()
			
			
	def compute_beltrami_on_scalar(self, f):
		for i in range(self.num_batches):
			tensorvertices, output_vertices, start, end = self.get_indexed_batch(i)
			
			beltrami_result = self.diffmod.laplace_beltrami_divgrad( out=output_vertices, wrt=tensorvertices, f= f(output_vertices) )
			
			self.all_beltrami_on_scalar[start:end] = beltrami_result.detach().numpy().copy()
			
	def compute_beltrami_H(self):
		for i in range(self.num_batches):
			tensorvertices, output_vertices, start, end = self.get_indexed_batch(i)
			
			normals = self.diffmod.compute_normals(out = output_vertices, wrt = tensorvertices)
			
			beltrami_result_x = self.diffmod.laplace_beltrami_divgrad( out=output_vertices, wrt=tensorvertices, f=output_vertices[:,0])
			beltrami_result_y = self.diffmod.laplace_beltrami_divgrad( out=output_vertices, wrt=tensorvertices, f=output_vertices[:,1])
			beltrami_result_z = self.diffmod.laplace_beltrami_divgrad( out=output_vertices, wrt=tensorvertices, f=output_vertices[:,2])
			beltrami_result = torch.stack((beltrami_result_x, beltrami_result_y, beltrami_result_z)).transpose(0,1)
			
			sign = -1*torch.sign( (beltrami_result * normals).sum(-1) ) 
			print('sign shape', sign.shape)
			print(sign)
			
			self.all_beltrami_H[start:end] = (0.5*sign*torch.linalg.norm(beltrami_result, dim=1) ).detach().numpy().copy()
