import os
import sys
import shutil
from normalise_mesh import normalise_mesh
import re


filepath = '/Users/romywilliamson/Documents/SphericalNS/'


name = sys.argv[1]
print('Processing the '+name+' mesh.')

def step0():
	print('This script automatically takes care of most of the setup for running an overfitting experiment.')
	print('You will need:')
	print('- a genus-0 triangle mesh stored at sns/data/[name].obj')
	print('- a spherical mesh embedding of this mesh, stored at sns/data/[name]_final_embedding.obj')
	print('In the paper, we use MultiRes sphere embedding from Schmidt et. al. \n Code here: https://github.com/patr-schm/surface-maps-via-adaptive-triangulations')
	

def step1():
	############## normalise mesh by bounding box ###############
	print('\nNormalising by area:')
	normalise_mesh(name, normalisation='B', rotate=False)
	print('Area normalisation done. \n')

def step2():
	pass
	########## Make a call to your chosen mesh parametrisation script, here,
	########## or leave this function empty if you already computed the spherical mesh embedding elsewhere.
	########## In the paper, we use MultiRes sphere embedding from Schmidt et. al. Code here: 
	########## https://github.com/patr-schm/surface-maps-via-adaptive-triangulations
	
	########## Once it is generated, the sphere-mesh .obj file should be located at
	########## sns/data/[name]_final_embedding.obj
	

def step3():
	########## copy embedding to temporary filename, ready for sphere process sample ###########
	shutil.copy( filepath+'sns/data/'+name+'_final_embedding.obj' ,
		      filepath+'sns/data/000temp_final_embedding.obj' )

	######################## make a pth file for the parametrisation ################################################
	print('\nMaking a pth file for the parametrisation.')
	os.system('python -m scripts.sphere_process_surface_sample --data '+filepath+'sns/data/'+name+'_nA.obj --square')
	print('Finished making pth file for the parametrisation. \n')



def step4():
	########################### write a json experiment file ######################################
	##### This step creates a new experiment json file, based on a template json file. ############
	
	print('\nWriting a new experiment json file.')
	with open(filepath+'sns/neural_surfaces-main/experiments/overfit/GENERIC.json') as generic_file:
		generic_json_string = generic_file.read()
	print(generic_json_string)

	
	specific_json_string = re.sub('XXX-NAME-XXX', name, generic_json_string, count=0, flags=0)
	with open(filepath+'sns/neural_surfaces-main/experiments/overfit/'+name+'.json', "w") as text_file:
	    text_file.write(specific_json_string)
	print('\nFinished writing a new experiment json file.')


def step5():
	print ('\nNow, to run the experiment you must simply use this command:')
	print('python -m mains.training experiments/overfit/'+name+'.json')
	
step0()
step1()
step2()
step3()
step4()
step5()