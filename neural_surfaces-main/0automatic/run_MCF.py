import numpy as np
import os
import sys
import shutil
from utils.normalise_mesh import normalise_mesh
import re
import torch
import json

from runners import MainRunner
from mains.experiment_configurator import ExperimentConfigurator
from differential import *

if len(sys.argv)>1:
        name = sys.argv[1]
        #name = sys.argv[2]
else:
        name = input('Input the name of the SNS. E.g. <<MAX10606>> .')
with open('experiment_configs/MCF/' + name + '.json') as f:
    config = json.load(f)

####### move  the intial weights and param to the correct place
os.system('cp ../data/SNS/'+name+'/param.pth ../data/MCF/'+name+'/current_param.pth')
os.system('cp ../data/SNS/'+name+'/weights.pth ../data/MCF/'+name+'/current_weights.pth')
#print('copied over weights and param from SNS folder')

diffmod = DifferentialModule()
modules_creator = ExperimentConfigurator()
runner = MainRunner('experiment_configs/MCF/SPIKE.json', modules_creator)## easier than sys.argv[1]
model = runner.get_model()


max_iter = config['params']['max_iter']
d = config['params']['flow_step_size']


for i in range(max_iter):
	os.system('cp ../data/MCF/'+name+'/current_weights.pth ../data/MCF/'+name+'/weights'+str(i)+'.pth')
	#load sample and update model
	sample = torch.load('../data/MCF/'+name+'/current_param.pth')
	weights = torch.load('../data/MCF/'+name+'/current_weights.pth', map_location=torch.device('cpu'))
	#weights = torch.load('../data/SNS/'+name+'/weights.pth', map_location=torch.device('cpu'))
	model = runner.get_model()
	model.load_state_dict(weights)
	model.eval()

	#compute H and n
	param = sample['param']
	tensorvertices = torch.Tensor(param)
	print(tensorvertices.dtype)
	tensorvertices.requires_grad=True
	output_vertices = model.forward(tensorvertices)
	normals = diffmod.compute_normals(out = output_vertices, wrt = tensorvertices)
	H,K,_,_,_,_,_ = diffmod.compute_curvature(out = output_vertices, wrt = tensorvertices, compute_principal_directions=False)
	#H = H.clip(min=-1e3, max=1e3) #prevent artefacts due to unstable high curvatures

	#update sample
	sample['points'] = (output_vertices - d*H.unsqueeze(-1)*normals).detach()
	sample['normals'] = normals.detach()
	print(tensorvertices)
	print(output_vertices.detach().numpy())
	torch.save(sample, '../data/MCF/'+name+'/current_param.pth')
	
	print('prepared iteration',i)      
	


	#free up some memory
	del param
	del tensorvertices
	del output_vertices
	del normals
	del H
	del K	
	del model

	#overfit to this sample
	os.system('python -m mains.training experiment_configs/MCF/'+name+'.json')	
	
























