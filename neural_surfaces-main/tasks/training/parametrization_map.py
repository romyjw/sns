
from runners import TrainRunner


class ParametrizationMapTrainRunner(TrainRunner):
    ## surface map optimnisation (one surface)

         
     
    def forward_model(self, batch, model, experiment):

        points2D_source = batch['source_points']
        #R               = batch['R']
        #t               = batch['t']
        #C_target        = batch['C_target']
        #C_source        = batch['C_source']

        #target_domain = batch['target_domain']

        points2D_source.requires_grad_(True)

        mapped_points, points3D_init, points3D_final = model(points2D_source)
        #points3D_target *= C_target
        #points3D_source *= C_source

        model_out = [mapped_points, points3D_init, points3D_final]
        loss, logs = experiment['loss'](mapped_points, points3D_init, points3D_final)

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
            
            points_mapped = model.forward_map(input_sphere_points, no_normalise=True)
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

