from eigenfunc.prepare_sample import sphere_prepare_global_sample
''' this function, borrowed from eigenfunc, makes a sample with these attributes
    sample['param'] , sample['points'] , sample['normals'] ,      
    sample['transpose_inverse_jacobians'] ,  sample['name'] , 
    sample['ortho_functions'] = None , sample['C'] = 1
    
    argument is just SNS_name
'''
import torch
import numpy as np

import trimesh
import pyrender
from visuals.visualisation_functions import *
import sys
import os



def visualise_pointcloud(vertices, colors):
	ptcloud_mesh = pyrender.Mesh.from_points(vertices, colors=colors)
	show_mesh_gui([ptcloud_mesh])#Display meshes.
	
def visualise_pointclouds(vertices_list, colors_list):
	rd_list = [pyrender.Mesh.from_points(vertices_list[i], colors=colors_list[i]) for i in range(len(colors_list))]
	show_mesh_gui(rd_list)#Display meshes.


	
visualise = True
	
SNS_name = sys.argv[1]
target_num_samples = sys.argv[2]
sample = sphere_prepare_global_sample(SNS_name, target_number_samples = target_num_samples)
points = sample['points'].detach().numpy() 
	
if visualise==True:
	visualise_pointcloud(points, np.array([0,0,0]))
	
torch.save(sample, '../data/deepsdf/'+SNS_name+'/SNSsample.pth')


################ go and call script in deepsdf repo to get sdf values ##############

os.system('cd ../../DeepSDF\n python -m scripts.reconstruct_from_latent '+SNS_name+'\n cd ../spherical2/neural_surfaces-main')

####################################################################################


new_sample = torch.load('../data/deepsdf/'+SNS_name+'/SNSsampleNew.pth')
sdf_gradients = new_sample['sdf_gradients'].detach()
sdf_values = new_sample['sdf_values'].detach()
normals = new_sample['normals'].detach()
new_points = new_sample['points'].detach().numpy()

colours = 0.5* normals.detach().numpy() + 0.5
#colours = np.clip( 0.5* sdf_gradients.detach().numpy() + 0.5, 0, 1)
#colours = (( 0.5* sdf_gradients + 0.5 ).T * (sdf_gradients.pow(2).sum(-1)<0.001)).T
#colours = np.array([1.0,1.0,1.0])


if visualise==True:
	print(new_points.shape, points.shape)
	#visualise_pointcloud(points, colours)
	visualise_pointcloud(new_points, colours)
	visualise_pointcloud(new_points, np.array([0.0,0,0]))
	visualise_pointclouds([points.squeeze(),new_points.squeeze()] , [np.array([0.0,0,0]), np.array([1.0,0,0])])
	

print('mean sdf gradient', sdf_gradients.pow(2).sum(-1).mean())
print('median sdf gradient', sdf_gradients.pow(2).sum(-1).median())
print('max sdf gradient', sdf_gradients.pow(2).sum(-1).max())
print('min sdf gradient', sdf_gradients.pow(2).sum(-1).min())
import matplotlib.pyplot as plt
plt.plot(sdf_gradients.pow(2).sum(-1))
plt.show()


plt.plot(sdf_values)
plt.show()






