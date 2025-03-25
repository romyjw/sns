import numpy as np
import torch

from utils.mesh import clean_mesh

import sys
import os

from runners import MainRunner
from mains.experiment_configurator import ExperimentConfigurator
import torch

from sample_preparation.eigenfunc import sample_prep_functions #might be wrong path

name = sys.argv[1]

#############

sample = sample_prep_functions.prepare_sample(name) #where the action happens!!! Generating samples.

################## add in ortho function samples, if any ######################
sample['ortho_functions'] = []


spherepoints = sample['param']
modules_creator = ExperimentConfigurator()
runner = MainRunner('experiments/eigenfunc/test.json', modules_creator)## easier than sys.argv[1]
model = runner.get_model()

if not os.path.exists('../data/eigenfunc/'+name+'/'):  # Check if the folder exists
        os.mkdir('../data/eigenfunc/'+name+'/')  # Create the folder if it doesn't exist

if not os.path.exists('../data/eigenfunc/'+name+'/orthoweights/'):  # Check if the folder exists
        os.mkdir('../data/eigenfunc/'+name+'/orthoweights/')  # Create the folder if it doesn't exist

for weightsfile in os.listdir('../data/eigenfunc/'+name+'/orthoweights/'):
	if weightsfile[-4:]=='.pth':
		weights = torch.load('../data/eigenfunc/'+name+'/orthoweights/'+weightsfile, map_location=torch.device('cpu'))
		
		model.load_state_dict(weights)
		model.eval()
		
		ortho_function = model(spherepoints).mean(-1) #I take the mean, because I am using a network with 3 outputs and it needs 1 output only
		sample['ortho_functions'].append(	ortho_function.detach())
	
#######################################################################

## save file as pth
if not os.path.exists('../data/eigenfunc/'+name):
	os.mkdir('../data/eigenfunc/'+name)
output_file = '../data/eigenfunc/'+name+ '/samples' + '.pth'
torch.save(sample, output_file)
print('Made the eigenfunc pth file, with ortho functions.')
