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
	
	
		

name_mesh_dict = {
					'SMALLTREE' : [('SMALLTREE', 'Analytic'),  ('SMALLTREE4', 'Coarse'), ('SMALLTREE5', 'Medium'), ('SMALLTREE6','Fine')],
					
					'SMALLBOBBLE' : [('SMALLBOBBLE', 'BobbleAnalytic'),  ('SMALLBOBBLE4', 'BobbleCoarse'), ('SMALLBOBBLE5', 'BobbleMedium'), ('SMALLBOBBLE6','BobbleFine')],
					
					'SMALLFLOWER' : [('SMALLFLOWER', 'FlowerAnalytic'),  ('SMALLFLOWER4', 'FlowerCoarse'), ('SMALLFLOWER5', 'FlowerMedium'), ('SMALLFLOWER6','FlowerFine')],
					
					'SMALLSPIKE' : [('SMALLSPIKE', 'SpikeAnalytic'),  ('SMALLSPIKE4', 'SpikeCoarse'), ('SMALLSPIKE5', 'SpikeMedium'), ('SMALLSPIKE6','SpikeFine')]
					}
		




colorstyles = {
    'gt': ('-', '#000000'),  # Black for ground truth

    'SNS': {
        'SMALLTREE4': ((0, (3, 1)), '#006400'),  # Dark green
        'SMALLTREE5': ((0, (3, 1)), '#32CD32'),  # Lime green
        'SMALLTREE6': ((0, (3, 1)), '#98FB98'),  # Pale green
        'SMALLTREE':  ('-', '#ff0000'),   # Red
        
        'SMALLBOBBLE4': ((0, (3, 1)), '#006400'),  # Dark green
        'SMALLBOBBLE5': ((0, (3, 1)), '#32CD32'),  # Lime green
        'SMALLBOBBLE6': ((0, (3, 1)), '#98FB98'),  # Pale green
        'SMALLBOBBLE':  ('-', '#ff0000'),   # Red
        
        'SMALLFLOWER4': ((0, (3, 1)), '#006400'),  # Dark green
        'SMALLFLOWER5': ((0, (3, 1)), '#32CD32'),  # Lime green
        'SMALLFLOWER6': ((0, (3, 1)), '#98FB98'),  # Pale green
        'SMALLFLOWER':  ('-', '#ff0000'),   # Red
        
        'SMALLSPIKE4': ((0, (3, 1)), '#006400'),  # Dark green
        'SMALLSPIKE5': ((0, (3, 1)), '#32CD32'),  # Lime green
        'SMALLSPIKE6': ((0, (3, 1)), '#98FB98'),  # Pale green
        'SMALLSPIKE':  ('-', '#ff0000'),   # Red
    },

    'i3d': ((0, (3,1,1,1)), '#800080'),  # Dark purple (i3d)
    
    'monge': ((0, (3,1,1,1)), '#9370DB'),  # Lighter lavender purple (monge)

    'NFGP': ((0, (3,1,1,1)), '#ff9900'),  # Bright orange-yellow
}






surf_name = sys.argv[-4]
show_analytic=True


quantity = sys.argv[-3] # normals, H or K
quantities = [quantity]

plot_type = sys.argv[-2] # error or dist

show = sys.argv[-1] # error or dist



methods = ['gt', 'i3d', 'monge', 'NFGP'] #don't include SNS here
#methods = ['gt', 'i3d', 'monge'] #don't include SNS here



from .evaluator_class import Anal_Surface_Evaluator

evaluator = Anal_Surface_Evaluator(surf_name)
evaluator.gen_points(start_N=200000, target_N=20000)



evaluator.evaluate_quantities()


################## GROUND TRUTH CURVE ############################


if plot_type=='dist':
	style, color = colorstyles['gt']
	evaluator.plot_distributions(quantities, ['gt'], style=style, color=color) #plot gt in solid black





################## OTHER METHODS CURVES ############################

for mesh_name, mesh_alias in name_mesh_dict[surf_name][-1:]: #just do highest res, to avoid clutter


	mesh_alias = ''#overwrite
	
	
	evaluator.mesh_name = mesh_name
	evaluator.load_surf_mesh()
	
	
	if 'monge' in methods:
	
	
		style, color = colorstyles['monge']
	
		
		evaluator.match_points()
		evaluator.monge_fitting_curvatures()
		
		if plot_type=='dist':
			evaluator.plot_distributions(quantities, ['monge'], extra_label=mesh_alias, style=style, color=color)
		elif plot_type=='error':
			evaluator.compute_errors(quantities, ['monge'])
			evaluator.plot_errors(quantities, ['monge'], extra_label=mesh_alias, style=style, color=color )
		
	if 'i3d' in methods:
		
		style, color = colorstyles['i3d']
	

		evaluator.i3d_curvatures()
		
		if plot_type=='dist':
			evaluator.plot_distributions(quantities, ['i3d'], extra_label=mesh_alias, style=style, color=color)
		elif plot_type=='error':
			evaluator.compute_errors(quantities, ['i3d'])
			evaluator.plot_errors(quantities, ['i3d'], extra_label=mesh_alias, style=style, color=color)
	
	
	if 'NFGP' in methods:
		
		style, color = colorstyles['NFGP']
	

		evaluator.NFGP_curvatures()
		
		if plot_type=='dist':
			evaluator.plot_distributions(quantities, ['NFGP'], extra_label=mesh_alias, style=style, color=color)
		elif plot_type=='error':
			evaluator.compute_errors(quantities, ['NFGP'])
			evaluator.plot_errors(quantities, ['NFGP'], extra_label=mesh_alias, style=style, color=color)
		
		
################## DIFFERENT RESOLUTION SNS CURVES ############################
		
evaluator.load_dense_sphere_mesh(level=8)

for SNS_name, SNS_alias in name_mesh_dict[surf_name]:


	style, color = colorstyles['SNS'][SNS_name]
	
	
	
	evaluator.SNS_name = SNS_name
	evaluator.load_SNS()
	evaluator.SNS_match_points()
	
	evaluator.SNS_curvatures()
	
	
	if plot_type=='dist':
		evaluator.plot_distributions(quantities, ['SNS'], extra_label=SNS_alias, style=style, color=color  )
	elif plot_type=='error':
		evaluator.compute_errors(quantities, ['SNS'])
		evaluator.plot_errors(quantities, ['SNS'], extra_label=SNS_alias, style=style, color=color )


plt.ylabel('Frequency Density')

################ hacky thing for reordering labels in the legend ##########

'''

# Get the handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

order = [i for i in range(len(handles))]
order[-3:] = [order[-1], order[-2], order[-3]]

ordered_handles = [handles[i] for i in order]
ordered_labels = [labels[i] for i in order]

# Create legend with custom order
plt.legend(ordered_handles, ordered_labels)

plt.show()

'''





plt.savefig('../data/plots/fri4oct/'+surf_name+quantity+plot_type+'.pdf', dpi=300 )
plt.savefig('../data/plots/fri4oct/'+surf_name+quantity+plot_type+'.png', dpi=300 )


if show=='show':
	plt.show()


