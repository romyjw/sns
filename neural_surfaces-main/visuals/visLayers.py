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



import matplotlib.pyplot as plt



## use this in most cases
modules_creator = ExperimentConfigurator()
runner = MainRunner('experiment_configs/overfit/ARMADILLO21622.json', modules_creator)## easier than sys.argv[1]
model = runner.get_model()

surf_name = sys.argv[-1]
write_ply=True
reject_points_for_PCD=True
target_number_samples=10000
level=6

weights = torch.load('../data/SNS/'+surf_name+'/weights.pth', map_location=torch.device('cpu'))



	

	
##################################################################################

model.load_state_dict(weights)
model.eval()

tm_sphere = trimesh.load('../data/analytic/sphere/sphere'+str(level)+'.obj')



vertices = tm_sphere.vertices
faces = tm_sphere.faces



'''
tm_surf = tm_sphere.copy()


all_output_vertices = model.forward(torch.tensor(vertices, dtype=torch.float32))

tm_surf.vertices = all_output_vertices.detach().numpy()

render_trimesh(tm_surf)
'''

circle_vertices = []
theta = 2*np.pi * torch.arange(0,1,0.001)
shells = []


n_shells=20
step = 0.02


fig,ax=plt.subplots(1,2)
for i in range(n_shells):
	circle_vertices = (1+i*step)*torch.stack([torch.cos(theta), torch.sin(theta), 0*torch.sin(theta)]).transpose(0,1)
	shell = model.forward(circle_vertices).detach().numpy()
	

	color='red'
	ax[0].plot(shell[:,0], shell[:,1], color=color, linewidth=0.5, alpha=0.5 - 0.5*i/n_shells)
	ax[1].plot(circle_vertices[:,0], circle_vertices[:,1], color=color, linewidth=0.5, alpha=0.5 - 0.5*i/n_shells)


for i in range(n_shells):
	circle_vertices = (1-i*step)*torch.stack([torch.cos(theta), torch.sin(theta), 0*torch.sin(theta)]).transpose(0,1)
	shell = model.forward(circle_vertices).detach().numpy()
	

	color='blue'
	ax[0].plot(shell[:,0], shell[:,1], color=color, linewidth=0.5, alpha=0.5 - 0.5*i/n_shells)
	ax[1].plot(circle_vertices[:,0], circle_vertices[:,1], color=color, linewidth=0.5, alpha=0.5 - 0.5*i/n_shells)


circle_vertices = torch.stack([torch.cos(theta), torch.sin(theta), 0*torch.sin(theta)]).transpose(0,1)
shell = model.forward(circle_vertices).detach().numpy()
ax[0].plot(shell[:,0], shell[:,1], color='black', linewidth=1, alpha=1.0 )
ax[1].plot(circle_vertices[:,0], circle_vertices[:,1], color='black', linewidth=1.0, alpha=1.0)

ax[0].set_xlim(-1.2,1.2)
ax[0].set_ylim(-1.2,1.2)
	
ax[0].set_aspect('equal')


ax[1].set_xlim(-1.2,1.2)
ax[1].set_ylim(-1.2,1.2)
	
ax[1].set_aspect('equal')


plt.show()








