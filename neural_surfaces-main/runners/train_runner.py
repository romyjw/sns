
from .generic_runner import GenericRunner
import numpy as np

class TrainRunner(GenericRunner):

    ## base function for training
    def run(self, batch, model, experiment, epoch):
        
        
        #print(experiment.keys())
        #input()
        ## call model and compute main losses
        #try:
        #    unit_norm_start_reg_param = experiment['RQloss']['params']['unit_norm_start_reg_param']
        #    unit_norm_current_reg_param =np.clip( unit_norm_start_reg_param * (1 - epoch/10000), 100.0, unit_norm_start_reg_param)
        #    experiment['RQloss']['params']['unit_norm_current_reg_param'] = unit_norm_current_reg_param        
        #except:
        #    pass 
        #try:
        #    ortho_start_reg_param = experiment['RQloss']['params']['ortho_start_reg_param']
        #    ortho_current_reg_param =np.clip( ortho_start_reg_param * (1 - epoch/10000), 1.0e3, ortho_start_reg_param)
        #    experiment['RQloss']['params']['ortho_current_reg_param'] = ortho_current_reg_param        
        #except:
        #    pass
        
        #try:
        #    sphere_start_reg_param = experiment['geodesic_loss']['params']['sphere_reg_param']
        #    sphere_current_reg_param = sphere_start_reg_param
        #    experiment['geodesic_loss']['params']['sphere_reg_param'] = sphere_current_reg_param  
        #except:
        #    pass
        #try:
        #    endpoint_start_reg_param = experiment['geodesic_loss']['params']['sphere_reg_param']
        #    endpoint_current_reg_param = endpoint_start_reg_param
        #    experiment['geodesic_loss']['params']['endpoint_reg_param'] = endpoint_current_reg_param      
        #except:
        #    pass


        model_out, loss, logs = self.forward_model(batch, model, experiment)
        #logs['unit_norm_current_reg_param'] = unit_norm_current_reg_param
        #logs['ortho_current_reg_param'] = ortho_current_reg_param
        #print('loss:',loss)
    
        ######## use only if tensorboard not working  ################
        if False:
                 with open('LossFile.txt', 'a'):
                        f.write(str(loss.detach().cpu().numpy()))
                        f.write('\n')
	################################################################

    
        ## compute regularization terms if any
        loss_reg, logs = self.regularizations(model, experiment, model_out, batch, logs)
        ## add regularization to loss
        loss += loss_reg
        ## update log of loss
        logs['loss'] = loss.detach()

        return loss, logs
