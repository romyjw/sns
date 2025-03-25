
import logging
from statistics import median
from tqdm import trange
import time

from .training_loop import TrainingLoop


class GradientTrainingLoop(TrainingLoop):
    ### used in Neural Surface Maps


    def compute_gradient(self):
        ## compute the median gradient for all parameters (median of mean, easy aggregation and does not clutter logging)
        gradient = []
        for _, params in self.model.named_parameters():
            if params.grad is not None:
                tmp_grad = params.grad.abs()
                gradient.append(tmp_grad.mean())

        return median(gradient)
    
    def loop(self):
        ## training loop
        for epoch in trange(self.num_epochs):
            epoch_start_time = time.time()
            #print('num_batches: ' , len(self.train_loader))
            for batch in self.train_loader:
                batch_start_time = time.time()
                
                part_start_time = time.time()
                self.zero_grads()
                zero_grads_time = time.time() - part_start_time
    
                part_start_time = time.time()
                batch = self.runner.move_to_device(batch) # move data to device
                move_to_device_time = time.time() - part_start_time
    
                part_start_time = time.time()
                loss, logs = self.runner.train_step(batch, epoch) # training iteration
                train_step_time = time.time() - part_start_time
    
                part_start_time = time.time()
                loss.backward()
                backward_time = time.time() - part_start_time
    
                part_start_time = time.time()
                self.optimize() # optimize
                optimize_time = time.time() - part_start_time
    
                part_start_time = time.time()
                self.log_train(logs) # log data
                log_train_time = time.time() - part_start_time
    
                batch_time = time.time() - batch_start_time
    
                # Print or log times for the batch
                #print(f"Batch time: {batch_time:.4f}s")
                #print(f"  zero_grads_time: {zero_grads_time:.4f}s")
                #print(f"  move_to_device_time: {move_to_device_time:.4f}s")
                #print(f"  train_step_time: {train_step_time:.4f}s")
                #print(f"  backward_time: {backward_time:.4f}s")
                #print(f"  optimize_time: {optimize_time:.4f}s")
                #print(f"  log_train_time: {log_train_time:.4f}s")
    
            part_start_time = time.time()
            self.checkpointing(epoch) # checkpoint
            checkpointing_time = time.time() - part_start_time
    
            part_start_time = time.time()
            self.scheduling() # scheduling
            scheduling_time = time.time() - part_start_time
    
            epoch_time = time.time() - epoch_start_time
    
            # Print or log times for the epoch
            #print(f"Epoch time: {epoch_time:.4f}s")
            #print(f"  checkpointing_time: {checkpointing_time:.4f}s")
            #print(f"  scheduling_time: {scheduling_time:.4f}s")
    
    
    '''
    def loop(self):
        
        print('I am traingin lop gradient loop')

        num_samples = len(self.train_loader)

        ## training loop
        for epoch in trange(self.num_epochs):

            grad = 0
            for batch in self.train_loader:

                converged = False

                self.zero_grads()

                batch = self.runner.move_to_device(batch)
                loss, logs = self.runner.train_step(batch, epoch)#changed12.04

                loss.backward()

                gradient = self.compute_gradient()
                grad += gradient

                self.optimize()

                self.log_train(logs)

                ## check if the gradient has vanished or below
                ## threshold then model has converged and can stop
                if gradient < self.grad_stop:
                    converged = True
                    break

            ## log gradient info once every epoch (avoid clutter)
            self.log_train({'gradient':grad/num_samples})

            self.checkpointing(epoch)

            self.scheduling()

            if converged:
                logging.info('Stopping!! Model has low gradient')
                break
        '''
