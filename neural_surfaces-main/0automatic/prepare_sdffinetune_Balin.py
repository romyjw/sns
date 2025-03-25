import os
import sys
import shutil
from normalise_mesh import normalise_mesh
import re
import torch


name = sys.argv[1]
target_num_points = sys.argv[2]
#name = sys.argv[2]

print('Processing the '+name+' mesh.')

########### define globals
sf = None
translation = None


def step0():
	print('This is the script for finetuning an SNS, using a deepsdf.')


def step1():
	########### deal with scaling/ translation? #################
	pass

def step2():
	print('\nSampling points on the SNS and projecting them to the isosurface.')
	
	os.system('python -m deepsdf.prepare_sample '+name)
	### sample gets saved in ../spherical2/data/deepsdf/'+SNS_name+'/SNSsampleNew.pth
	
	############# Call the sampling/projection process and save pth file of samples.

def step3():
	
	print('Finished making pth file for the parametrisation. \n')



def step4():
	########################### write a json experiment file ######################################
	print('\nWriting a new experiment json file.')
	with open('experiments/sdffinetune/GENERIC.json') as generic_file:
		generic_json_string = generic_file.read()
	print(generic_json_string)

	
	specific_json_string = re.sub('XXX-NAME-XXX', name, generic_json_string, count=0, flags=0)
	with open('experiments/sdffinetune/'+name+'.json', "w") as text_file:
	    text_file.write(specific_json_string)
	print('\nFinished writing a new experiment json file.')

def step5():
	input('Please take this opportunity to check that the experiment json file is as you wish and make any edits if required.')
	
	

def step6():
	print ('\nNow, to run the experiment you must simply use this command, here on Balin:')
	print('python -m mains.training experiments/sdffinetune/'+name+'.json')
	
step0()
step1()
step2()
step3()
step4()
#step5()
step6()


###shutil.copy('/Users/romywilliamson/Documents/SphericalNS/spherical2/neural_surfaces-main/experiments/overfit/'+name+'_nA_overfit_experiment.json',   '/Users/romywilliamson/ethaetfh.json'   )






















