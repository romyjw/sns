import numpy as np
import torch



import sys
import os

from runners import MainRunner
from mains.experiment_configurator import ExperimentConfigurator
import torch


from sample_preparation.geodesic import sample_prep_functions

name = sys.argv[1]

#############

sample = sample_prep_functions.prepare_sample(name) #where the action happens!!! Generating samples.
########### for geodesics it makes sense to just use evenly spaced samples.


	
#######################################################################

## save file as pth
if not os.path.exists('../data/geodesic/'+name):
	os.mkdir('../data/geodesic/'+name)
output_file = '../data/geodesic/'+name+ '/samples' + '.pth'
torch.save(sample, output_file)
print('Made the geodesic pth file')
