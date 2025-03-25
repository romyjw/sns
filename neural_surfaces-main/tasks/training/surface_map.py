from runners import TrainRunner 
from utils.calculate_volume import calculate_volume

import torch


class SurfaceMapTrainRunner(TrainRunner):
    ## surface to surface map trainer
    


    def g(self, x):
        ### min of 0 at 1, asymptote at 0 < c < 1, increase linearly-ish after 1
        c = 0.7
        return 1.0/(x - c) + (x-1.0)*((1.0-c)**(-2.0)) - 1.0/(1.0-c)
    
    def k(self, x, d):
        ### min of 0 at 1, asymptotes at 1 +/- d
        return -1.0/((1.0+d - x)*(1.0-d - x)) - 1.0/(d**2)
   
    def k_grad(self, x, d):
        ### gradient of  the k function
        return (2*x - 2.0)*(1.0 - d**2 + x**2 - 2.0*x )**(-2.0)

    def k_mod(self, x, d, s):
        ### make linear after certain point
        x0 = 1.0 - d*s
        x1 = 1.0 + d*s
 
        left_linear_part = self.k_grad(x0,d )*(x - x0) + self.k(x0, d)
        right_linear_part = self.k_grad(x1,d )*(x - x1) + self.k(x1, d)
        return self.k(x,d) * torch.logical_and(x>=x0, x<=x1) + left_linear_part*(x<x0) + right_linear_part*(x>x1)      

    def w(self, x, s):
        return torch.cosh(s*(x-1.0)) - 1.0 


    def forward_model(self, batch, model, experiment):

        points2D_source = batch['source_points']
        R               = batch['R']
        #t               = batch['t']
        #C_target        = batch['C_target']
        #C_source        = batch['C_source']

        target_domain = batch['target_domain']

        points2D_source.requires_grad_(True)

        points3D_target, points2D_target, points3D_source = model(points2D_source, R)
        #points3D_target *= C_target
        #points3D_source *= C_source

        model_out = [points3D_target, points2D_target, points2D_source, points3D_source]
        loss, logs = experiment['loss'](points3D_target, points2D_target, points2D_source, points3D_source, target_domain)

        return model_out, loss, logs


    def regularizations(self, model, experiment, predictions, batch, logs):
        loss = 0.0

        ##if experiment['loss'].reg_boundary > 0.0:
        ##    source_boundary = batch['boundary']
        ##    R               = batch['R']
        ##    #t               = batch['t']

        ##    source_boundary_mapped = model.forward_map(source_boundary, R)

        ##    loss_boundary, _ = experiment['loss'].boundary_loss(source_boundary_mapped, source_boundary)

        ##    logs['loss_boundary'] = loss_boundary.detach()
        ##    loss += experiment['loss'].reg_boundary * loss_boundary


        if experiment['loss'].reg_landmarks > 0.0:
            target_landmarks = batch['target_landmarks']   [:, :]
            landmarks        = batch['landmarks']          [:, :]
            R                = batch['R']
            
            
            landmarks_mapped = model.forward_map(landmarks, alignment_sig = R)
            
                       
            
            ## Compute landmark loss in 2D
            loss_lands_Sph = (target_landmarks - landmarks_mapped).pow(2).sum(-1).mean()
            logs['loss_lands_Sph'] = loss_lands_Sph.detach()





            ## Compute landmark loss in 3D
            landmarks3D_mapped = model.target_surface(landmarks_mapped)
            landmarks3D_target = model.target_surface(target_landmarks)
            sq_norms = (landmarks3D_target - landmarks3D_mapped).pow(2).sum(-1)

            loss_lands_R3 = sq_norms.mean()
            logs['loss_lands_R3'] = loss_lands_R3.detach()
            
            for i in range(sq_norms.shape[0]):
                logs['distance_land_'+str(i)] = sq_norms[i].sqrt()

            loss_lands = loss_lands_R3
            ### loss_lands = 0.5*(loss_lands_Sph + loss_lands_R3) ### set landmark loss to be the average of the MSE on the sphere and on the target shape
           
            #would be good to log both if possible.
            
            loss += experiment['loss'].reg_landmarks * loss_lands
            logs['loss_landmarks'] = loss_lands.detach()
        



        if experiment['loss'].reg_spherical > 0.0:
            #target_landmarks = batch['target_landmarks']
            #landmarks        = batch['landmarks']
            input_sphere_points = batch['source_points']
            R                = batch['R']
            #t                = batch['t']

            # landmarks_mapped = model.forward_map(landmarks, R)
            #print('I am SPHERE loss being called.')
            #print('batch keys',batch.keys())
            #print('source points shape',batch['source_points'].shape)
            
            points_mapped = model.forward_map(input_sphere_points, R, no_normalise=True)
            sq_norms = points_mapped.pow(2).sum(-1)

            # loss_spherical = self.k_mod(sq_norms**0.5, 0.1, 0.999).mean()
            loss_spherical = self.k(sq_norms**0.5, 0.1).mean()
            # loss_spherical = self.w(sq_norms**0.5, 150.0).mean()

            loss += experiment['loss'].reg_spherical * loss_spherical
            logs['loss_spherical'] = loss_spherical.detach()
        
        
        if experiment['loss'].reg_volume > 0.0:
            PI = 3.14159265
            
            icosphere_vertices = batch['icosphere'][vertices] ## figure out how to access these later
            icosphere_faces = batch['icosphere'][faces]
            
            
            R   = batch['R']
            icosphere_vertices_mapped = model.forward_map(input_sphere_points, R)
            
            volume = calculate_volume(icosphere_vertices_mapped, icosphere_faces)
            
            loss_volume = self.k(volume / ( PI * 4.0/3.0 ))

            loss += experiment['loss'].reg_volume * loss_volume
            logs['loss_volume'] = loss_volume.detach()
        
        
        
        
        
        return loss, logs
