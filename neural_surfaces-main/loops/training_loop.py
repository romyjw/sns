
import logging
import torch
from tqdm import trange


class TrainingLoop():

    def __init__(self, runner, num_epochs, **kwargs):
        self.runner = runner
        self.num_epochs = num_epochs
        self.__dict__.update(kwargs)


    def configure_run(self):
        ## get objects needed to run the experiment
        self.train_loader  = self.runner.train_loader()
        self.model         = self.runner.get_model()
        self.optimizers    = self.runner.get_optimizers()
        self.schedulers    = self.runner.get_schedulers()
        self.logger        = self.runner.get_logger()


    def run(self):

        self.configure_run()

        self.interrupted = False

        ## move model to device and update the runner
        self.model = self.runner.move_to_device(self.model)
        self.runner.model = self.model

        ## notify the training is starting
        with torch.no_grad():
            self.runner.train_starts()

        ## training loop
        try:
            self.loop()

            ## notify end training
            with torch.no_grad():
                self.runner.train_ends()

        except KeyboardInterrupt:
            logging.info('Detected KeyboardInterrupt, attempting graceful shutdown...')

            # user could press ctrl+c many times... only shutdown once
            if not self.interrupted:
                self.interrupted = True
                with torch.no_grad():
                    self.runner.train_ends()

    '''
    def loop(self):

        ## training loop
        for epoch in trange(self.num_epochs):
            for batch in self.train_loader:

                self.zero_grads()

                batch = self.runner.move_to_device(batch) # move data to device
                loss, logs = self.runner.train_step(batch, epoch) # training iteration   #changed12.04

                loss.backward()
                self.optimize() # optimize

                self.log_train(logs) # log data

            self.checkpointing(epoch) # checkpoint
            self.scheduling() # scheduling
        '''
        
        
    def loop(self):
        ## training loop
        for epoch in trange(self.num_epochs):
            epoch_start_time = time.time()
            print('num_batches: ' , len(self.train_loader))
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
                print(f"Batch time: {batch_time:.4f}s")
                print(f"  zero_grads_time: {zero_grads_time:.4f}s")
                print(f"  move_to_device_time: {move_to_device_time:.4f}s")
                print(f"  train_step_time: {train_step_time:.4f}s")
                print(f"  backward_time: {backward_time:.4f}s")
                print(f"  optimize_time: {optimize_time:.4f}s")
                print(f"  log_train_time: {log_train_time:.4f}s")
    
            part_start_time = time.time()
            self.checkpointing(epoch) # checkpoint
            checkpointing_time = time.time() - part_start_time
    
            part_start_time = time.time()
            self.scheduling() # scheduling
            scheduling_time = time.time() - part_start_time
    
            epoch_time = time.time() - epoch_start_time
    
            # Print or log times for the epoch
            print(f"Epoch time: {epoch_time:.4f}s")
            print(f"  checkpointing_time: {checkpointing_time:.4f}s")
            print(f"  scheduling_time: {scheduling_time:.4f}s")
    
    


    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def optimize(self):
        for opt in self.optimizers:
            opt.step()

    def scheduling(self):
        for sch in self.schedulers:
            sch.step()

    def log_train(self, logs):
        self.logger.log_data(logs)

    def checkpointing(self, epoch):
        if (epoch+1) % self.checkpoint_epoch == 0:
            with torch.no_grad():
                self.runner.checkpoint(epoch)
        
