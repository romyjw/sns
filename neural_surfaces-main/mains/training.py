import torch
torch.autograd.set_detect_anomaly(True)
import sys



import torch.multiprocessing as mp #added


from runners import MainRunner

from .experiment_configurator import ExperimentConfigurator

mp.set_start_method('fork', force=True) #added



if __name__ == '__main__': #this is required for running on a mac
	## use this in most cases
	modules_creator = ExperimentConfigurator()
	runner = MainRunner(sys.argv[1], modules_creator)
	runner.run_loop()
