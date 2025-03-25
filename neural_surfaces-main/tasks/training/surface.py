
from differential import DifferentialModule
from runners import TrainRunner


class SurfaceTrainer(TrainRunner, DifferentialModule):
    ## trainer for NSM

    def forward_model(self, batch, model, experiment):
        param  = batch['param']
        gt     = batch['gt']
        #c = batch['c']
        #print('c is', c)

        if experiment['loss'].reg_normals > 0.0:
            param.requires_grad_(True)

        points     = model(param)
        loss, logs = experiment['loss'](points, gt)
        logs['loss_distance'] = loss.detach()
        logs['log_loss_distance'] = loss.detach().log()


        return points, loss, logs

    def regularizations(self, model, experiment, predictions, batch, logs):
        points = predictions
        loss = 0.0

        if experiment['loss'].reg_normals > 0.0:
            normals = batch['normals'].view(-1, 3)
            #print(batch['mask'].shape)
            mask    = batch['mask'].view(-1)

            pred_normals = self.compute_normals(out=points, wrt=batch['param']).view(-1, 3)
            #loss_norm, logs_norm = experiment['loss'](pred_normals[mask], normals[mask])
            loss_norm, logs_norm = experiment['loss'](pred_normals, normals)
            logs['loss_norm'] = logs_norm['loss']
            loss += experiment['loss'].reg_normals * loss_norm

        return loss, logs
