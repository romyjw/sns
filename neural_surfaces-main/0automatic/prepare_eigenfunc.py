import os
import sys
import shutil
from utils.normalise_mesh import normalise_mesh
import re


if len(sys.argv)>1:
	name = sys.argv[1]
	#name = sys.argv[2]
else:
	name = input('Input the name of the neural surface (e.g. MAX10606) or SPHERE.')
if name=='':
	name='SPHERE'

identifier = input('Input a unique experiment identifier.')
#eigNumber = int (input('Input the index for the eigenvector that you want to optimise.') )

def step0():
	print ('This is the script for preparing samples to approximate integration over the surface.')





def step1():
	
	

	######################## make samples ################################################
	print('\nMaking a pth file with samples in it.')
	#os.system('python -m scripts.sphere_process_surface_sample --data ../data/'+name+'_nB.obj --square')
	#os.system('python -m scripts.eigenfunc_sphere_process_surface_sample '+name)
	os.system('python -m sample_preparation.eigenfunc.process_eigenfunc_sample '+name)
	
	print('Finished making pth file for the parametrisation. \n')



def step2():
	########################### write a json experiment file ######################################
	print('\nWriting a new experiment json file.')
	with open('experiment_configs/eigenfunc/GENERIC.json') as generic_file:
		generic_json_string = generic_file.read()
	print(generic_json_string)

	
	specific_json_string = re.sub('XXX-NAME-XXX', name, generic_json_string, count=0, flags=0)
	specific_json_string = re.sub('XXX-IDENTIFIER-XXX', identifier, specific_json_string, count=0, flags=0)
	
	with open('experiment_configs/eigenfunc/'+name+'.json', "w") as text_file:
	    text_file.write(specific_json_string)
	print('\nFinished writing a new experiment json file.')

def step3():
	input('Please take this opportunity to check that the experiment json file is as you wish and make any edits if required.')
	
	

def step4():
	print ('\nNow, to run the experiment you must simply use this command, here on Balin:')
	print('python -m mains.training experiment_configs/eigenfunc/'+name+'.json')
	
step0()
step1()
step2()
step3()
step4()
























