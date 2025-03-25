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
		
		
		self.quantities = {'H':dict(), 'K':dict(), 'normals':dict()}
		self.errors = {'H':dict(), 'K':dict(), 'normals':dict()}
		
		
		self.scale, _ = normalise_mesh('mesh'+str(self.mesh_level), normalisation=self.normalisation_type, rotate=False, directory='../data/analytic/'+self.surf_name+'/')
		#self.scale = 1.0
		
		
		
		
		
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
        'scale': self.scale,
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
		#dir11=local_context['dir11']
		#dir12=local_context['dir12']
		#dir13=local_context['dir13']
		#dir21=local_context['dir21']
		#dir22=local_context['dir22']
		#dir23=local_context['dir23']
		
		
		
		H *=-1
		#print(H)
			
		self.quantities['H']['gt'] = H
		self.quantities['K']['gt'] = K
		##### construct normals array ####
		self.quantities['normals']['gt'] = replace_nans( np.stack([n1,n2,n3]).transpose() )
		#self.quantities['dir1']['gt'] = replace_nans ( np.stack([dir11,dir12,dir13]).transpose() )
		#self.quantities['dir2']['gt'] = replace_nans ( np.stack([dir21,dir22,dir23]).transpose() )
		
		#### correct the order of min and max curvature directions ####
		#self.quantities['dir1']['gt'], self.quantities['dir2']['gt'] = (self.quantities['dir2']['gt'], self.quantities['dir1']['gt'])
		
		#### normalize the directions ####
		#self.quantities['dir1']['gt'] = replace_nans( self.quantities['dir1']['gt'].transpose() / np.linalg.norm(self.quantities['dir1']['gt'], axis=1) ).transpose()
		#self.quantities['dir2']['gt'] = replace_nans( self.quantities['dir2']['gt'].transpose() / np.linalg.norm(self.quantities['dir2']['gt'], axis=1) ).transpose()
		
		## fix signs
		#frame = (np.stack([self.quantities['dir1']['gt'], self.quantities['dir2']['gt'], self.quantities['normals']['gt']])).transpose((1,0,2))
		#print(frame.shape)
		#signs = np.linalg.det(frame)
		#self.quantities['dir1']['gt'] = (self.quantities['dir1']['gt'].T*signs.T).T
			

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
		
		
		
		
	'''def load_sphere_mesh(self, meshpath):
		self.tm_sphere = trimesh.load(meshpath)'''
	
	def load_surf_mesh(self):
		self.meshpath = '../data/analytic/'+self.surf_name+'/mesh6_nB.obj'
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
		
		
		#weights = torch.load('../data/SNS/'+self.surf_name+'/weights.pth', map_location=torch.device('cpu'))
		
		
		
		
		####### to use MESH-based fitting, not analytic fitting
		
		weights = torch.load('../data/SNS/'+self.SNS_name+'/weights.pth', map_location=torch.device('cpu'))
	
		model.load_state_dict(weights)
		model.eval()
		
		
		
		
		batches_diff_quant = bdq(self.sph_points, model, diffmod, batch_size, scaling = self.scale)

		batches_diff_quant.compute_SNS()
		batches_diff_quant.compute_curvature()

		batches_diff_quant.compute_normals()
		batches_diff_quant.compute_directions()
		
			
		############# get the info from the batches computation #######
		self.quantities['H']['SNS'] = batches_diff_quant.all_H
		self.quantities['K']['SNS'] = batches_diff_quant.all_K
		self.quantities['normals']['SNS'] = batches_diff_quant.all_normals
		#self.quantities['SNS_directions'] = batches_diff_quant.all_directions
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

		
		
		
		
		#print(self.quantities['H']['monge'].shape)
		
		
	
	def NFGP_curvatures(self):
		np.save('../../NFGP/data/'+self.surf_name+'/sample_points.npy',(self.surf_points  ) )
		
		
		command = 'python ../../NFGP/train.py ../../NFGP/configs/curv/'+self.surf_name+'.yaml'
		os.system(command)
		
		self.quantities['H']['NFGP'] = np.load('../data/evaluation/'+self.surf_name+'/NFGP_H.npy').squeeze()
		self.quantities['K']['NFGP'] = np.load('../data/evaluation/'+self.surf_name+'/NFGP_K.npy').squeeze()
		self.quantities['normals']['NFGP'] = np.load('../data/evaluation/'+self.surf_name+'/NFGP_normals.npy').squeeze()
	
	
	
	
	def i3d_curvatures(self):
		np.save('../../i3d/data/'+self.surf_name+'/sample_points.npy',self.surf_points)
		command = 'python ../../i3d/tools/'+self.surf_name+'_estimate_sample_curvatures.py'
		os.system(command)
		
		self.quantities['H']['i3d'] = np.load('../../i3d/results/sample_points_H.npy').squeeze()
		self.quantities['K']['i3d'] = np.load('../../i3d/results/sample_points_K.npy').squeeze()
		self.quantities['normals']['i3d'] = np.load('../../i3d/results/sample_points_normals.npy').squeeze()
	
	
	
	
	def compute_errors(self, quantities, methods):
		
		for quantity in quantities:
			
			if not quantity in ['normals', 'dir1', 'dir2']:
				for method in methods:
			
					self.errors[quantity][method] = ( self.quantities[quantity]['gt'] - self.quantities[quantity][method])**2
				
			else:
				for method in methods:
			
					self.errors[quantity][method] = np.arccos( (self.quantities[quantity]['gt'] * self.quantities[quantity][method]).sum(-1) ) * 180/np.pi
					

	

	
	
	
	def plot_distributions(self, quantities, methods, extra_label=''):
		
		for quantity in quantities:
			for method in methods:
				style = '-.'
				if method=='gt':
					style = 'k-'
				draw_histogram_curve(self.quantities[quantity][method], range=[-50,20], bins=30, n=5000, label=method+ '_' + quantity + '_'+self.surf_name+'_'+extra_label, style=style)
		
		plt.legend()
		
				
		#if show_analytic==True:
		#	draw_histogram_curve(self.quantities['H']['gt'], range=[-50,20], bins=30, n=5000, label='analytic H', style='k-')	
		#draw_histogram_curve(self.quantities['H']['monge'], range=[-50,20], bins=50, n=500, label='Monge Fitted H', style='r-')	
		#draw_histogram_curve(self.quantities['H']['i3d'], range=[-50,20], bins=50, n=500, label='i3d H', style='g-')
		#draw_histogram_curve(self.quantities['H']['NFGP'], range=[-50,20], bins=50, n=500, label='NFGP H', style='m-')	
		
		
		
	
		
	
	def plot_errors(self, quantities, methods, extra_label = ''):
		
		for quantity in quantities:
			for method in methods:
				

				if method=='gt':
					pass
				else:
				
					if not quantity in ['normals', 'dir1', 'dir2']:
						x_range = [0,1]
					else:
						x_range = [0,90]
						
					draw_histogram_curve(self.errors[quantity][method], range=x_range, bins=10, n=500, label=method+ '_' + quantity + '_'+self.surf_name+'_'+extra_label+'_error', style='-.')
		
			plt.legend()
			
			
			if not quantity in ['normals', 'dir1', 'dir2']:
				plt.xlabel('Error (SSE)')
			else:
				plt.xlabel('Angle Error (in degrees)')
			
		

		

name_mesh_dict = {'SPIKE' : ['S-analytic-SNS',  'S-remesh4', 'S-remesh5', 'S-remesh6'],
					'LAUNDRY' : ['L-analytic-SNS', 'L-remesh4', 'L-remesh5', 'L-remesh6'],
					'FLOWER' : ['F-analytic-SNS',  'F-remesh4', 'F-remesh5', 'F-remesh6'],
					'TREE' : ['T-analytic-SNS',  'T-remesh4', 'T-remesh5', 'T-remesh6']
					}
		


surf_name = sys.argv[-1]
show_analytic=True

#quantities = ['H']
quantities = ['normals']
methods = ['gt','i3d', 'NFGP', 'monge']

evaluator = Anal_Surface_Evaluator(surf_name)
evaluator.gen_points(start_N=20000, target_N=2000)
evaluator.load_surf_mesh()


evaluator.evaluate_quantities()

evaluator.i3d_curvatures()

evaluator.NFGP_curvatures()

evaluator.match_points()
evaluator.monge_fitting_curvatures()
	
evaluator.compute_errors(quantities, methods)	
	
#evaluator.plot_distributions(quantities, methods)
evaluator.plot_errors(quantities, methods)


for SNS_name in name_mesh_dict[surf_name]:
	evaluator.SNS_name = SNS_name
	
	evaluator.SNS_curvatures()
	



	#evaluator.show_distributions(show_analytic=show_analytic)
	#show_analytic=False
	
	evaluator.compute_errors(quantities, ['SNS'])
	evaluator.plot_errors(quantities, ['SNS'], extra_label=SNS_name)
	#evaluator.plot_distributions(quantities, ['SNS'], extra_label=SNS_name )


plt.show()




