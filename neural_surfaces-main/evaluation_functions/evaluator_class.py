


from utils import rejection_sampling
from utils.formula_converter import *
from utils.plotting import *
import numpy as np
import trimesh
from scipy.spatial import KDTree
import sys
import re

#SNS stuff
from differential import *
from differential.batches_diff_quant import batches_diff_quant as bdq

from runners import MainRunner
from mains.experiment_configurator import ExperimentConfigurator
import torch

#i3d stuff
import shutil
import os
import subprocess

#sampling
from utils.rejection_sampling import rejection_sampling

#mesh
from utils.normalise_mesh import normalise_mesh


def replace_nans(arr, value=0.0):
	arr[np.isnan(arr)] = value
	return arr
	
	
		
class Anal_Surface_Evaluator:
	def __init__(self, surf_name):
		
		
		self.surf_name = surf_name
		
		
		
		
		self.mesh_level = 6
		self.normalisation_type = 'B'
		
		
		self.quantities = {'geom': dict(), 'H':dict(), 'K':dict(), 'normals':dict(), 'dir1':dict(), 'dir2':dict(), 'mincurvdir':dict()}
		self.errors = {'geom': dict(), 'H':dict(), 'K':dict(), 'normals':dict(), 'mincurvdir':dict()}
		
	
		
		with open('../data/analytic/'+surf_name+'/formula.txt') as rf:
			formula = rf.read()
			
		self.formula = formula
		
		with open('../data/analytic/'+surf_name+'/analytic_result.txt') as rf:
			matlab_equations = rf.read()
		
		self.numpy_equations = matlab_to_numpy(matlab_equations)
	
	def evaluate_quantities(self):
		
		theta = np.arccos(self.sph_points[:,2].clip(-1,1))
		phi = np.arctan2(self.sph_points[:,1], self.sph_points[:,0])
		
		
		#print(theta)
		#print(phi)
		
		local_context = {
		'pi': 3.14159265,
        'scale': 1.0,
        'theta': theta,
        'phi': phi,
        'np':np}
		
		
		for statement in self.numpy_equations:
			#print(statement)
			#print()
			 exec(statement, {}, local_context)# compute/define H,K,n1,n2,n3, dir11, dir12, dir13, dir21, dir22, dir23
			 
			 
		H=local_context['H']
		K=local_context['K']
		n1=local_context['n1']
		n2=local_context['n2']
		n3=local_context['n3']
		dir11=local_context['dir11']
		dir12=local_context['dir12']
		dir13=local_context['dir13']
		dir21=local_context['dir21']
		dir22=local_context['dir22']
		dir23=local_context['dir23']
		
		
		
		H *=-1
		#print(H)
		
		
		self.quantities['geom']['gt'] = self.surf_points
		self.quantities['H']['gt'] = H
		self.quantities['K']['gt'] = K
		##### construct normals array ####
		self.quantities['normals']['gt'] = replace_nans( np.stack([n1,n2,n3]).transpose() )
		self.quantities['dir1']['gt'] = replace_nans ( np.stack([dir11,dir12,dir13]).transpose() )
		self.quantities['dir2']['gt'] = replace_nans ( np.stack([dir21,dir22,dir23]).transpose() )
		
		#### correct the order of min and max curvature directions ####
		self.quantities['dir1']['gt'], self.quantities['dir2']['gt'] = (self.quantities['dir2']['gt'], self.quantities['dir1']['gt'])
		
		#### normalize the directions ####
		self.quantities['dir1']['gt'] = replace_nans( self.quantities['dir1']['gt'].transpose() / np.linalg.norm(self.quantities['dir1']['gt'], axis=1) ).transpose()
		self.quantities['dir2']['gt'] = replace_nans( self.quantities['dir2']['gt'].transpose() / np.linalg.norm(self.quantities['dir2']['gt'], axis=1) ).transpose()
		
		## fix signs
		frame = (np.stack([self.quantities['dir1']['gt'], self.quantities['dir2']['gt'], self.quantities['normals']['gt']])).transpose((1,0,2))
		#print(frame.shape)
		signs = np.linalg.det(frame)
		self.quantities['dir1']['gt'] = (self.quantities['dir1']['gt'].T*signs.T).T
			
		self.quantities['mincurvdir']['gt'] = self.quantities['dir2']['gt']
		
		
		
		
		
	def gen_points(self, target_N, start_N):#make uniform samples on (scaled) analytic surface
	
		#gen points on sphere
		#reject points based on FFF
		#find theta, phi
		#put points through analytic function
		
		P = np.random.randn(3,start_N)
		P = P/np.sqrt(np.einsum('ij,ij->j',P,P))
		P = np.transpose(P)#now N x 3
		
		
		U = np.arccos(P[:,2])###### arccos(z)
		V = np.arctan2(P[:,1], P[:,0])
		
		

		
		local_context = {
		'pi': 3.14159265,
        'scale': 1.0,
        'theta': U,
        'phi': V,
        'np':np}
		
		
		for statement in self.numpy_equations:
			#print(statement)
			#print()
			 exec(statement, {}, local_context)# compute/define FFF, SFF etc
			 
			 
		E=local_context['E']
		F=local_context['F']
		G=local_context['G']
		
		area_density = np.sqrt(E*G-F**2)
		
		P, area_density, keeping_iis = rejection_sampling(P, area_density, target_N) 
		
		U = U[keeping_iis]
		V = V[keeping_iis]
	
		
		
		self.sph_points = P
		self.U=U
		self.V=V
		self.surf_points = eval(self.formula)
		
		
		
		
	'''def load_sphere_mesh(self, meshpath):
		self.tm_sphere = trimesh.load(meshpath)'''
	
	def load_dense_sphere_mesh(self, level=8):

		dense_sphere_meshpath = '../data/analytic/sphere/sphere'+str(level)+'.obj'
		self.dense_tm_sphere = trimesh.load(dense_sphere_meshpath)
	
	def load_rendering_mesh(self, level=6):

		rendering_sphere_meshpath = '../data/analytic/sphere/sphere'+str(level)+'.obj'
		self.rendering_tm_sphere = trimesh.load(rendering_sphere_meshpath)
		
		self.sph_points = self.rendering_tm_sphere.vertices
		
		P = self.sph_points
		U = np.arccos(P[:,2])###### arccos(z)
		V = np.arctan2(P[:,1], P[:,0])
		self.surf_points = eval(self.formula)
		
		self.rendering_tm_surf = self.rendering_tm_sphere
		self.rendering_tm_surf.vertices = self.surf_points
		
		
		

	def load_surf_mesh(self):
		self.meshpath = '../data/'+self.mesh_name+'.obj'
		self.tm_surf = trimesh.load(self.meshpath)

	def match_points(self):  # Find closest mesh vertices
	    vertices = self.tm_surf.vertices
	    
	    #print(vertices)
	    
	    # Create a KDTree from the vertices
	    kdtree = KDTree(vertices)
	    
	    # Query the KDTree for the closest vertex to each point in self.surf_points
	    distances, indices = kdtree.query(self.surf_points)
	    
	    # indices is a list of the closest vertex index for each point in self.surf_points
	    self.vertex_indices = indices.tolist()
	    
	    #print('distances')
	    #print(self.surf_points - self.tm_surf.vertices[self.vertex_indices])
	    
	
	
	def SNS_match_points(self, batch_size=20000):  # Find closest mesh vertices
		print('matching SNS points')
		
		dense_sph_vertices = self.dense_tm_sphere.vertices
		batches_diff_quant = bdq(dense_sph_vertices, self.model, self.diffmod, batch_size, scaling = 1.0)
		
		batches_diff_quant.compute_SNS()
		
		dense_surf_points = batches_diff_quant.all_output_vertices
		
		# Create a KDTree from the vertices
		kdtree = KDTree(dense_surf_points)
		
		# Query the KDTree for the closest vertex to each point in self.surf_points
		distances, indices = kdtree.query(self.surf_points)
		
		# indices is a list of the closest vertex index for each point in self.surf_points
		self.SNS_vertex_indices = indices.tolist()
		
		self.SNS_sphere_points = dense_sph_vertices[self.SNS_vertex_indices]

		print('finished matching SNS points')
	
	def load_SNS(self):
		self.diffmod = DifferentialModule()
		self.modules_creator = ExperimentConfigurator()
		self.runner = MainRunner('experiment_configs/overfit/ARMADILLO21622.json', self.modules_creator)#armadillo is just generic
		self.model = self.runner.get_model()
		
		
		#weights = torch.load('../data/SNS/'+self.surf_name+'/weights.pth', map_location=torch.device('cpu'))
		
		
		
		
		####### to use MESH-based fitting, not analytic fitting
		
		weights = torch.load('../data/SNS/'+self.SNS_name+'/weights.pth', map_location=torch.device('cpu'))
	
		self.model.load_state_dict(weights)
		self.model.eval()

	
	
	def SNS_curvatures(self, batch_size=2000):
		#from differential import batches_diff_quant
		
				
		
		
		
		batches_diff_quant = bdq(self.SNS_sphere_points, self.model, self.diffmod, batch_size, scaling = 1.0)

		batches_diff_quant.compute_SNS()
		batches_diff_quant.compute_curvature()

		batches_diff_quant.compute_normals()
		batches_diff_quant.compute_directions()
		
			
		############# get the info from the batches computation #######
		self.quantities['geom']['SNS'] = batches_diff_quant.all_output_vertices
		
		self.quantities['H']['SNS'] = batches_diff_quant.all_H
		self.quantities['K']['SNS'] = batches_diff_quant.all_K
		self.quantities['normals']['SNS'] = batches_diff_quant.all_normals
		self.quantities['mincurvdir']['SNS'] = batches_diff_quant.all_directions[0]
		
		#self.quantities['SNS_points'] = batches_diff_quant.all_output_vertices
		
	
	def monge_fitting_curvatures(self):
		
		self.tm_surf.export('../../CGAL-mesh-processing/build/current_mesh.off')
		
		
		# Path to the executable's directory
		exec_dir = '../../CGAL-mesh-processing/build'

		# Run the executable in its own directory
		subprocess.run('./curvatures', cwd=exec_dir, check=True)
		
		#command = '../../CGAL-mesh-processing/build/curvatures'
		#os.system(command)
		
		vertex_only_H = -1.0 * np.loadtxt('../../CGAL-mesh-processing/build/mean_curvature.txt')
		self.quantities['H']['monge'] = vertex_only_H[self.vertex_indices]
		
		vertex_only_K = np.loadtxt('../../CGAL-mesh-processing/build/gauss_curvature.txt')
		self.quantities['K']['monge'] = vertex_only_K[self.vertex_indices]
		
		vertex_only_normals = np.loadtxt('../../CGAL-mesh-processing/build/monge_normals.txt')
		self.quantities['normals']['monge'] = vertex_only_normals[self.vertex_indices]
		
		
		vertex_only_mincurvdir = np.loadtxt('../../CGAL-mesh-processing/build/mincurvdir.txt')
		self.quantities['mincurvdir']['monge'] = vertex_only_mincurvdir[self.vertex_indices]
		
		self.quantities['geom']['monge'] = None

		
		#print(self.quantities['H']['monge'].shape)
		
		
	
	def NFGP_curvatures(self):
		np.save('../../NFGP/data/'+self.surf_name+'/sample_points.npy',(self.surf_points  ) )
		
		
		command = 'python ../../NFGP/train.py ../../NFGP/configs/curv/'+self.surf_name+'.yaml'
		os.system(command)
		
		self.quantities['H']['NFGP'] = np.load('../data/evaluation/'+self.surf_name+'/NFGP_H.npy').squeeze()
		self.quantities['K']['NFGP'] = np.load('../data/evaluation/'+self.surf_name+'/NFGP_K.npy').squeeze()
		self.quantities['normals']['NFGP'] = np.load('../data/evaluation/'+self.surf_name+'/NFGP_normals.npy').squeeze()
		self.quantities['geom']['NFGP'] = None
	
	
	
	def i3d_curvatures(self):
		np.save('../../i3d/data/'+self.mesh_name+'/sample_points.npy',self.surf_points)
		
		
		
		print('\nWriting a new i3d curvature script.')
		with open('../../i3d/tools/GENERIC_estimate_sample_curvatures.py', 'r') as generic_file:
			generic_string = generic_file.read()

	
		
		specific_string = re.sub('XXX-GENERIC-XXX', self.mesh_name, generic_string, count=0, flags=0)
		
		
		with open('../../i3d/tools/'+self.mesh_name+'_estimate_sample_curvatures.py', 'w') as text_file:
		    text_file.write(specific_string)
		print('\nFinished writing a new i3d curvature script.')
		

		
		
		command = 'python ../../i3d/tools/'+self.mesh_name+'_estimate_sample_curvatures.py'
		os.system(command)
		
		self.quantities['H']['i3d'] = np.load('../../i3d/results/sample_points_H.npy').squeeze()
		self.quantities['K']['i3d'] = np.load('../../i3d/results/sample_points_K.npy').squeeze()
		self.quantities['normals']['i3d'] = np.load('../../i3d/results/sample_points_normals.npy').squeeze()
		
		##### warning: I am correcting for a bug in i3d, I think by switching max and min directions
		self.quantities['mincurvdir']['i3d'] = np.load('../../i3d/results/sample_points_maxcurvdir.npy').squeeze()	
		#self.quantities['maxcurvdir']['i3d'] = np.load('../../i3d/results/sample_points_mincurvdir.npy').squeeze()	
		
		self.quantities['geom']['i3d'] = None
	
	
	def compute_errors(self, quantities, methods):
		
		
		
		
		for quantity in quantities:
			
			if quantity in ['H', 'K']:
				for method in methods:
					if self.quantities[quantity][method] is not None:
			
						self.errors[quantity][method] = abs( self.quantities[quantity]['gt'] - self.quantities[quantity][method])
				
			elif quantity in ['normals', 'mincurvdir']:
				for method in methods:
					if self.quantities[quantity][method] is not None:
						self.errors[quantity][method] = np.arccos( (np.abs(self.quantities[quantity]['gt'] * self.quantities[quantity][method]).sum(-1) ) ) * 180/np.pi
			
			elif quantity in ['geom']:
				for method in methods:
					if self.quantities[quantity][method] is not None:
						self.errors[quantity][method] = ((( self.quantities[quantity]['gt'] - self.quantities[quantity][method])**2).sum(-1))**(0.5)

	

	
	
	
	def plot_distributions(self, quantities, methods, extra_label='', style='-', color=None):
		
		for quantity in quantities:
			for method in methods:
				if self.quantities[quantity][method] is not None:

					draw_histogram_curve(self.quantities[quantity][method], range=[-50,20], bins=30, n=50000, label=method + '_'+extra_label, style=style, color=color)
			
			plt.xlabel(quantity)
		plt.legend()
		
				

	
		
	
	def plot_errors(self, quantities, methods, extra_label = '', style = '-', color=None):
		
		for quantity in quantities:
			for method in methods:
				
				

				if method=='gt':
					pass
				else:
					if self.quantities[quantity][method] is not None:
				
						if quantity in [ 'K' ]:
							x_range = [0,8]
						if quantity in ['H' ]:
							x_range = [0,8]
						elif quantity in ['geom']:
							x_range = [0,0.025]
						elif quantity in ['normals', 'mincurvdir']:
							x_range = [0,20]
							
						draw_histogram_curve(self.errors[quantity][method], range=x_range, bins=20, n=50000, label=method + '_'+extra_label+'_error', style=style, color=color)
		
			plt.legend()
			
			
			
			if quantity == 'normals':
				plt.xlabel('Normals Error (in degrees)')
			elif quantity == 'mincurvdir':
				plt.xlabel('Minimum Curvature Direction Error (in degrees)')
			elif quantity == 'K':
				plt.xlabel('Gauss Curvature (K) Error (Absolute Difference)')
			elif quantity == 'H':
				plt.xlabel('Mean Curvature (H) Error (Absolute Difference)')
			elif quantity == 'geom':
				plt.xlabel('Fitting Error (Absolute Difference)')
			
			
			'''
			
			if not quantity in ['normals', 'mincurvdir']:
				plt.xlabel('Error (Absolute Difference)')
			else:
				plt.xlabel('Angle Error (in degrees)')
			'''
		

