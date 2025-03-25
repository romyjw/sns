
from differential import DifferentialModule
from runners import TrainRunner
import numpy as np






class EigenfuncTrainer(TrainRunner, DifferentialModule):
    ## trainer for eigenfunctions
    
           
        
    def forward_model(self, batch, model, experiment):
        param  = batch['param']
        normals = batch['normals']
        transpose_inverse_jacobians = batch['transpose_inverse_jacobians']
        ortho_functions = batch['ortho_functions']        
        


        #print('param,normal alignment',(param*normals).sum(-1).mean())
        #print('transpose_inverse_jacobians',transpose_inverse_jacobians)
        logs = {}
        #c = batch['c']

        #print ('TIJ shape in tasks/training/eigenfunc',transpose_inverse_jacobians.shape)
        area = 4.0*np.pi

        param.requires_grad_(True) #because we will need to differentiate wrt param, to find the gradient.

        f_values    = model(param)

        #print('f values shape', f_values.shape)
        
        f_values = f_values.mean(-1)
        
        
        dirichlet_energy = self.dirichlet_energy(f_values.unsqueeze(-1), param, transpose_inverse_jacobians, normals)
                
        logs['dirichlet_energy'] = dirichlet_energy.detach()
        
        rayleigh_quotient = self.rayleigh_quotient(f_values.unsqueeze(-1), param, transpose_inverse_jacobians, normals)
                
        logs['rayleigh_quotient'] = rayleigh_quotient.detach()

        #unit norm regularisation
        unit_norm_reg = (area*f_values.pow(2).mean() - 1.0)**2
        logs['unit_norm_reg'] = unit_norm_reg.detach()
        
        
        full_ortho_reg = 0.0
        
        zeroth_ortho_reg = f_values.mean().pow(2)
        logs['0th ortho reg'] = zeroth_ortho_reg
        full_ortho_reg += zeroth_ortho_reg

        for i in range(len(ortho_functions)):
            current_ortho_reg = (f_values*ortho_functions[i]).mean().pow(2)
            logs['ortho reg'+str(i+1)] = current_ortho_reg
            full_ortho_reg += current_ortho_reg
            
        logs['full ortho reg'] = full_ortho_reg
            
        
        
        
        #loss = dirichlet_energy +  1e4* unit_norm_reg + 1e4*full_ortho_reg

        unit_norm_reg_param = experiment['RQloss']['params']['unit_norm_current_reg_param']
        ortho_reg_param = experiment['RQloss']['params']['ortho_current_reg_param']

        loss = rayleigh_quotient + unit_norm_reg_param*unit_norm_reg + ortho_reg_param*full_ortho_reg
        
        
        return f_values, loss, logs
    
    
    def dirichlet_energy(self, f_values, param, transpose_inverse_jacobians=None, normals=None):
        #print ('TIJ shape in dirichlet energy function',transpose_inverse_jacobians.shape)
        DfDsph = self.gradient(out = f_values, wrt = param).squeeze().unsqueeze(-1) # it's 4048 x 1 x 3 if I don't squeeze. 


        DfDsurf =  ( transpose_inverse_jacobians @ DfDsph).squeeze()  #When the surface is just the sphere, the transpose inverse jacobians are all 3x3 identity matrices.
        #DfDsurf = DfDsph.squeeze() 
                
        coeffs = (DfDsurf*normals).sum(-1)
        covariant_grad = DfDsurf - (coeffs *  normals.T).T

        #dA = 4.0*np.pi/f_values.shape[0]
        #dirichlet_energy = (covariant_grad @ covariant_grad.T).sum()*dA
        
        #dA = 4.0*np.pi/f_values.shape[0]
        dirichlet_energy = (covariant_grad.pow(2).sum(-1)).mean()*4.0*np.pi
        
        #loss = dirichlet_energy #+ norm_regularisation
        return dirichlet_energy
        
    def rayleigh_quotient(self, f_values, param, transpose_inverse_jacobians=None, normals=None):
        dirichlet_energy = self.dirichlet_energy(f_values, param, transpose_inverse_jacobians, normals)
        l2norm_sq = ( 4.0*np.pi*(f_values.pow(2)).mean() )
        
        return dirichlet_energy/l2norm_sq
        
        
    def regularizations(self, model, experiment, predictions, batch, logs):
        points = predictions
        loss = 0.0

        return loss, logs

