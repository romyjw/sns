
import logging
import torch

import models
import datasets
import optimizers
import tasks
import loggers
import losses
import schedulers

#def ab_lambda_replacement(a,b):
#    return a

class ExperimentConfigurator():

    def __init__(self):
        configurator = {}
        configurator['datasets']   = datasets.create
        configurator['models']     = models.create
        configurator['optimizers'] = optimizers.create
        configurator['schedulers'] = schedulers.create
        configurator['loss']       = losses.create
        configurator['tasks']      = tasks.create
        configurator['logging']    = loggers.create
        #configurator['loop']       = ab_lambda_replacement
        
        self.configurator = configurator


    def create_experiment_modules(self, config):
        #does this just copy over the config data?
        experiment = {}
        if 'RQloss' in config.keys():
            experiment['RQloss']       = config['RQloss']
        
        if 'geodesic_loss' in config.keys():
            experiment['geodesic_loss']       = config['geodesic_loss']
        
        if 'sns_path' in config.keys():
            experiment['sns_path']       = config['sns_path']



        for key, value in config.items():
            if key not in self.configurator:
                logging.info(f'Skipping key {key}')
                continue
            experiment[key] = self.configurator[key](value, experiment)

        return experiment
