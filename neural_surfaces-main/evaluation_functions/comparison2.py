from utils import rejection_sampling
from utils.formula_converter import *
from utils.plotting import *
import numpy as np
import trimesh
from scipy.spatial import KDTree
import sys

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
		if self.surf_name=='REMESH6':
			self.normalisation_type = 'A'
		
		
		
		self.scale, _ = normalise_mesh(self.surf_name+'.obj', normalisation=self.normalisation_type, rotate=False, directory='../data/')
		
		
		

			

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
        'scale': self.scale,
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
		self.surf_points = self.scale * eval(self.formula)
		
		
		#fix the sampling later
		
		
	'''def load_sphere_mesh(self, meshpath):
		self.tm_sphere = trimesh.load(meshpath)'''
	
	def load_surf_mesh(self):
		self.meshpath = '../data/analytic/'+self.surf_name+'/mesh6_nA.obj'
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
		
	
	
	
	
	def SNS_curvatures(self, batch_size=2000):
		#from differential import batches_diff_quant
		
		diffmod = DifferentialModule()
		modules_creator = ExperimentConfigurator()
		runner = MainRunner('experiment_configs/overfit/ARMADILLO21622.json', modules_creator)#armadillo is just generic
		model = runner.get_model()
		
		
		weights = torch.load('../data/SNS/'+self.surf_name+'/weights.pth', map_location=torch.device('cpu'))
	
		model.load_state_dict(weights)
		model.eval()
		
		
		
		
		batches_diff_quant = bdq(self.sph_points, model, diffmod, batch_size, scaling = self.scale)

		batches_diff_quant.compute_SNS()
		batches_diff_quant.compute_curvature()

		batches_diff_quant.compute_normals()
		batches_diff_quant.compute_directions()
		
			
		############# get the info from the batches computation #######
		self.SNS_H = batches_diff_quant.all_H
		self.SNS_K = batches_diff_quant.all_K
		self.SNS_normals = batches_diff_quant.all_normals
		self.SNS_directions = batches_diff_quant.all_directions
		self.SNS_points = batches_diff_quant.all_output_vertices
		
	
	def monge_fitting_curvatures(self):
		
		self.tm_surf.export('../../CGAL-mesh-processing/build/current_mesh.off')
		
		
		# Path to the executable's directory
		exec_dir = '../../CGAL-mesh-processing/build'

		# Run the executable in its own directory
		subprocess.run('./curvatures', cwd=exec_dir, check=True)
		
		#command = '../../CGAL-mesh-processing/build/curvatures'
		#os.system(command)
		
		vertex_only_H = -1.0 * np.loadtxt('../../CGAL-mesh-processing/build/mean_curvature.txt')
		self.monge_fitted_H = vertex_only_H[self.vertex_indices]
		#print(self.monge_fitted_H.shape)
		
		
	
	def NFGP_curvatures(self):
		np.save('../../NFGP/data/'+self.surf_name+'/sample_points.npy',(self.surf_points  ) )
		
		
		
		
		
		command = 'python ../../NFGP/train.py ../../NFGP/configs/curv/'+self.surf_name+'.yaml'
		os.system(command)
		
		self.NFGP_H = np.load('../data/evaluation/'+self.surf_name+'/NFGP_H.npy').squeeze()
	
	def i3d_curvatures(self):
		np.save('../../i3d/data/'+self.surf_name+'/sample_points.npy',self.surf_points)
		command = 'python ../../i3d/tools/'+self.surf_name+'_estimate_sample_curvatures.py'
		os.system(command)
		
		self.i3d_H = np.load('../../i3d/results/sample_points_H.npy').squeeze()
		
	def evaluate(self):

		monge_error = ( self.H - self.monge_fitted_H)**2
		SNS_error = ( self.H - self.SNS_H)**2
		i3d_error = ( self.H - self.i3d_H)**2
		NFGP_error = ( self.H - self.NFGP_H)**2
		
		
		draw_histogram_curve(monge_error, range=[0,1], bins=200, n=500, label='Monge Fitting H error', style='r-.')	
		draw_histogram_curve(i3d_error, range=[0,1], bins=200, n=500, label='i3d H error', style='g-.')
		draw_histogram_curve(NFGP_error, range=[0,1], bins=200, n=500, label='NFGP H error', style='m-.')
		draw_histogram_curve(SNS_error, range=[0,1], bins=200, n=500, label='SNS H error', style='b-.')
		
		plt.legend()
		plt.show()
		
		
		
	
		draw_histogram_curve(self.H, range=[-50,20], bins=50, n=500, label='analytic H', style='k-')	
		draw_histogram_curve(self.monge_fitted_H, range=[-50,20], bins=50, n=500, label='Monge Fitted H', style='r-')	
		draw_histogram_curve(self.i3d_H, range=[-50,20], bins=50, n=500, label='i3d H', style='g-')
		draw_histogram_curve(self.NFGP_H, range=[-50,20], bins=50, n=500, label='NFGP H', style='m-')	
		draw_histogram_curve(self.SNS_H, range=[-50,20], bins=50, n=500, label='SNS H', style='b-')
		plt.legend()
		plt.show()
		
		
		

surf_name = sys.argv[-1]

evaluator = Anal_Surface_Evaluator(surf_name)
evaluator.gen_points(start_N=20000, target_N=2000)
evaluator.evaluate_quantities()

evaluator.load_surf_mesh()

evaluator.SNS_curvatures()
evaluator.i3d_curvatures()

evaluator.NFGP_curvatures()

evaluator.match_points()
evaluator.monge_fitting_curvatures()

evaluator.evaluate()




