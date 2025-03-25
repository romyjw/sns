from visuals.helpers.visualisation_functions import *
from visuals.helpers import rd_helper



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

from utils.custom_ply_writing import *




def replace_nans(arr, value=0.0):
	arr[np.isnan(arr)] = value
	return arr
	
	

name_mesh_dict = {
					'SMALLTREE' : [('SMALLTREE', 'TreeAnalytic'),  ('SMALLTREE4', 'TreeCoarse'), ('SMALLTREE5', 'TreeMedium'), ('SMALLTREE6','TreeFine')]
					}
	
	
name_mesh_dict = {
					'SMALLTREE' : [('SMALLTREE', 'TreeAnalytic'),  ('SMALLTREE4', 'TreeCoarse'), ('SMALLTREE5', 'TreeMedium'), ('SMALLTREE6','TreeFine')]
					}

error_cmap = plt.get_cmap('gist_yarg')


surf_name = sys.argv[-2]


quantities = [ 'H', 'K', 'mincurvdir', 'normals']

show = sys.argv[-1] 






from .evaluator_class import Anal_Surface_Evaluator





evaluator = Anal_Surface_Evaluator(surf_name)
#evaluator.gen_points(start_N=20000, target_N=2000)

evaluator.load_rendering_mesh(level=7)
evaluator.load_dense_sphere_mesh(level=8)


evaluator.evaluate_quantities()



		
################## DIFFERENT RESOLUTION SNS CURVES ############################
		

for SNS_name, SNS_alias in name_mesh_dict[surf_name]:


	
	
	evaluator.SNS_name = SNS_name
	
	evaluator.load_SNS()
	evaluator.SNS_match_points()
	
	evaluator.sph_points = evaluator.SNS_sphere_points
	
	evaluator.SNS_curvatures()
	
	evaluator.compute_errors(quantities, ['SNS'])
	
	evaluator.errors['geom']['SNS'] = (( evaluator.quantities['geom']['SNS']    -evaluator.surf_points   )**2).sum(-1)**(0.5)
	
	errordict=dict()
	
	for quantity in ['geom'] + quantities:
		
	
		evaluator.colours = error_cmap(evaluator.errors[quantity]['SNS'])
	
		if show=='show':
			print('showing the ', quantity, ' error for ', SNS_alias)
			render_trimesh(evaluator.rendering_tm_surf, evaluator.colours)
		
		
		

		
	torch.save(evaluator.errors, '../data/plots/'+SNS_name+'_errors.pth')
	evaluator.rendering_tm_surf.export('../data/plots/'+SNS_name+'.obj')

	
	
	


