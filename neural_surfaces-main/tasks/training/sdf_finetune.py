from differential import DifferentialModule
from runners import TrainRunner
import torch


class SDF_finetuneTrainer(TrainRunner, DifferentialModule):
    ## trainer for eigenfunctions
    
           
        
    def forward_model(self, batch, SNSmodel, SDFmodel, experiment):
        param  = batch['param'] #sphere points
        normals = batch['normals']
        
        latent_vec  =  batch['latent_vec']        


        #print('param,normal alignment',(param*normals).sum(-1).mean())
        #print('transpose_inverse_jacobians',transpose_inverse_jacobians)
        logs = {}
        #c = batch['c']

        #print ('TIJ shape in tasks/training/eigenfunc',transpose_inverse_jacobians.shape)
        area = 4.0*np.pi

        param.requires_grad_(True) #because we will need to differentiate wrt param, to find the gradient.

        pred_points    = SNSmodel(param)
        latent_vec_with_points = torch.stack(( latent_vec.repeat(), pred_points )) #placeholder
        sdf_values = SDFmodel(latent_vec_with_points)
        
        
        distance_loss = sdf_values.pow(2).sum(-1) #L2 loss for now
                
        logs['distance_loss'] = distance_loss.detach()
        

        loss = distance_loss
        
        
        return pred_points, loss, logs
    
        

    def regularizations(self, model, experiment, predictions, batch, logs):
        points = predictions
        loss = 0.0
        # the regularisations happen directly in the forward model now

        return loss, logs
