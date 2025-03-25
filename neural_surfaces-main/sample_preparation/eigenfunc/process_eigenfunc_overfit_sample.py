import numpy as np
import torch

import sys
import os

from runners import MainRunner
from mains.experiment_configurator import ExperimentConfigurator
import torch

from eigenfunc import sample_prep_functions

name = sys.argv[1]
eigNumber = int(sys.argv[2])

#############

sample = prepare_sample.sphere_prepare_eigenfunc_overfit(name, eigNumber) # Generating samples. In this case, samples are at mesh vertices.

#######################################################################

## save file as pth
if not os.path.exists('../data/eigenfunc/'+name):
	os.mkdir('../data/eigenfunc/'+name)
output_file = '../data/eigenfunc/'+name+ '/overfit_samples' +str(eigNumber)+ '.pth'
torch.save(sample, output_file)
print('Made the eigenfunc pth file for the eigenfunc overfitting stage.')
