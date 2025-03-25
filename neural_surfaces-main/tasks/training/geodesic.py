
from differential import DifferentialModule
from runners import TrainRunner
import numpy as np

import torch




class GeodesicTrainer(TrainRunner, DifferentialModule):
    ## trainer for eigenfunctions
    
    #def k(self, x, d):
        ### min of 0 at 1, asymptotes at 1 +/- d
    #    return -1.0/((1.0+d - x)*(1.0-d - x)) - 1.0/(d**2)
           
        
    def forward_model(self, batch, model, experiment):
        
        
        #sns_path = experiment['sns_path']
        #SNS_model = experiment['models']['sns_model'].create()
        #SNS_model.load_state_dict(torch.load(sns_path))
        param  = batch['param']     
        


        #print('param,normal alignment',(param*normals).sum(-1).mean())
        #print('transpose_inverse_jacobians',transpose_inverse_jacobians)
        logs = {}

        

        param.requires_grad_(True) #because we will need to differentiate wrt param, to find the gradient.
        
        device = param.device
     
        tgt_startpoint = torch.Tensor(experiment['geodesic_loss']['sphere_startpoint']).to(device)
        tgt_endpoint = torch.Tensor(experiment['geodesic_loss']['sphere_endpoint']).to(device)
        #print(tgt_startpoint, tgt_endpoint)



        surface_points, sphere_points    = model.forward(param, tgt_startpoint, tgt_endpoint)
        #surface_points = SNS_model(sphere_points)        
        
        dirichlet_energy = self.dirichlet_energy(surface_points, param)
                
        logs['dirichlet_energy'] = dirichlet_energy.detach()
        
        
       
        _, startpoint = model(torch.Tensor([[0]]).to(device),  tgt_startpoint, tgt_endpoint) #do endpoint loss on the sphere, for now
        _, endpoint = model(torch.Tensor([[1]]).to(device),  tgt_startpoint, tgt_endpoint)
       

        print(tgt_startpoint,startpoint, tgt_endpoint,endpoint)
        endpoint_reg = ( startpoint - tgt_startpoint ).pow(2).sum() +(endpoint - tgt_endpoint ).pow(2).sum()
        logs['endpoint_reg'] = endpoint_reg.detach()

        #sphere regularisation
        #sq_norms = sphere_points.pow(2).sum(-1)
    
        
        #sphere_reg = self.k(sq_norms**0.5, 0.1).mean()

        #logs['sphere_reg'] = sphere_reg.detach()
        

        #keep regularisation fixed for now
        endpoint_reg_coeff = experiment['geodesic_loss']['params']['endpoint_reg_param']
        #sphere_reg_coeff = experiment['geodesic_loss']['params']['sphere_current_reg_param']

        #loss = dirichlet_energy + endpoint_reg_coeff * endpoint_reg + sphere_reg_coeff * sphere_reg
        loss = dirichlet_energy + endpoint_reg_coeff * endpoint_reg        
        print('dirichlet energy:', dirichlet_energy)
        print('startpoint', startpoint)
        print('endpoint', endpoint)

        return surface_points, loss, logs
    
    
    def dirichlet_energy(self, surface_points, param):
        
        
        DgammaDt = self.gradient(out = surface_points, wrt = param).squeeze() # it's 4048 x 1 x 3 if I don't squeeze. 
        print(DgammaDt.shape)

        dirichlet_energy = (DgammaDt.pow(2).sum(-1)).mean()
        
        return dirichlet_energy
        
           
    def regularizations(self, model, experiment, predictions, batch, logs):
        points = predictions
        loss = 0.0

        return loss, logs

