import sys
from runners import MainRunner
from mains.experiment_configurator import ExperimentConfigurator
import torch

import numpy as np
import trimesh#to load mesh
import igl

import pyrender

from .helpers.visualisation_functions import *
from .helpers import rd_helper


from .helpers.subdiv import *
from torch import autograd as Grad


from differential import *

import copy

surf_name = 'MAX10606'

modules_creator = ExperimentConfigurator()
runner = MainRunner('experiment_configs/geodesic/MAX10606.json', modules_creator)
model = runner.get_model()

diffmod = DifferentialModule()



######################## options #################################


weights = torch.load('../data/geodesic/'+surf_name+'/linearinit1/model3699.pth', map_location=torch.device('cpu'))

print(weights.keys())
model.load_state_dict(weights)
model.eval()


input_points = torch.linspace(0, 1, steps=100).unsqueeze(1)
print(input_points.shape)

tgt_startpoint = torch.tensor([[1.0,0.0,0.0]])
tgt_endpoint = torch.tensor([[0.0,1.0,0.0]])

output = model.forward(input_points, tgt_startpoint, tgt_endpoint)

#print(output)

output_points = output[0].squeeze().detach().numpy()

print(output_points.shape)

#print(output_points)




marker_tm = trimesh.load('../data/analytic/sphere/sphere1.obj')
marker_tm.vertices = 0.01 * marker_tm.vertices

marker_tm.visual.vertex_colors = np.array([1.0,0.0,0.0])

n_markers = output_points.shape[0]
print(n_markers)
marker_tms = [copy.deepcopy(marker_tm) for i in range(n_markers)]
for i in range(n_markers):
	marker_tms[i].vertices = marker_tms[i].vertices + output_points[i,:]
	
	#print(output_points[i,:])


surf_tm = trimesh.load('../data/icosphere_MAX106066.obj')

marker_tms.append(surf_tm)


render_trimeshes(marker_tms)

render_trimesh(marker_tms[0])



