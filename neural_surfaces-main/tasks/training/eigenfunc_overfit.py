
from differential import DifferentialModule
from runners import TrainRunner
import numpy as np



class EigenfuncOverfitTrainer(TrainRunner, DifferentialModule):
    ## trainer for eigenfunctions
    
           
        
    def forward_model(self, batch, model, experiment):
        param  = batch['param']
        discrete_eigenfunc = batch['discrete_eigenfunc']      
        
        logs = {}
        
        param.requires_grad_(True) #because we will need to differentiate wrt param, to find the gradient.

        f_values    = model(param).mean(-1) #I take the mean, because I have been forced to use a network with 3 inputs and 3 outputs but there should only be 1 output.
        
        mse_loss = (f_values - discrete_eigenfunc).pow(2).mean()
        print('mse_loss',mse_loss)
        logs['eigenfunc mse_overfit_loss'] = mse_loss.detach()

        loss = mse_loss
        return f_values, loss, logs
    
    
    
    def regularizations(self, model, experiment, predictions, batch, logs):
        points = predictions
        loss = 0.0
        # the regularisations happen directly in the forward model now

        return loss, logs

