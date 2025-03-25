
import datetime
import logging
import os

from .experiment_runner import ExperimentRunner


class MainRunner(ExperimentRunner):
    ## more advanced version of the Generic runner

    def __init__(self, config, modules_creator):
        super().__init__(config, modules_creator)
        print(self.config.keys())
        self.checkpoint_dir = self.config['checkpointing']['base_path'] + self.config['checkpointing']['identifier']#self.config['checkpointing']['folder']
        
        
        if not os.path.isdir(self.config['checkpointing']['base_path']):
            os.mkdir(self.config['checkpointing']['base_path'])
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.isdir(self.checkpoint_dir+'/init'):
            os.mkdir(self.checkpoint_dir+'/init')
        if not os.path.isdir(self.checkpoint_dir+'/init/models'):
            os.mkdir(self.checkpoint_dir+'/init/models')
        if not os.path.isdir(self.checkpoint_dir+'/models'):
            os.mkdir(self.checkpoint_dir+'/models')
        ### also make the folders if needed
        ### include identifier if needed
        
        


    def train_starts(self):

        ## check what kind of libraries are there and remove checkpointing that cannot be used
        self.check_all_imports()

        ## if the train runner need to do something at startup call it
        train_fun = getattr(self.experiment['tasks']['train'], "train_starts", None)
        if callable(train_fun):
            self.experiment['tasks']['train'].train_starts(self.model, self.experiment, self.checkpoint_dir)

        ## same for checkpoint runner
        checkp_fun = getattr(self.experiment['tasks']['checkpoint'], "train_starts", None)
        if callable(checkp_fun):
            self.experiment['tasks']['checkpoint'].train_starts(self.model, self.experiment, self.checkpoint_dir)

        ## record time of start
        logging.info('Training starts')
        self.start_time = datetime.datetime.now()
        with open(self.checkpoint_dir+'/training_info.txt', 'w') as file:
            file.write('start time: '+str(self.start_time))


    def train_ends(self):
        ## record time of end
        self.end_time = datetime.datetime.now()
        self.training_time = self.end_time - self.start_time
       
        logging.info('Training ends')

        ## checkpoint experiment with full report
        ckpt_info = self.CKPTWrapper()
        #ckpt_info.generate_report = True
        ckpt_info.training_time   = self.training_time
        ckpt_info.checkpoint_dir  = self.checkpoint_dir
        ckpt_info.epoch           = int(1.0e10) #just a big number to show that it's finished
        
        self.experiment['tasks']['checkpoint'].run(self.model, self.experiment, ckpt_info)
        with open(self.checkpoint_dir+'/training_info.txt','a') as end_file:
            end_file.write('Training ended.')
            end_file.write(str(self.training_time)+' total time\n')
            end_file.write('start: '+str(self.start_time)+' end: '+str(self.end_time))
        ## render everything if possible
        # self.render()


    def checkpoint(self, epoch):
        ## normal checkpointing
        ckpt_info = self.CKPTWrapper()
        ckpt_info.checkpoint_dir = self.checkpoint_dir
        ckpt_info.epoch          = epoch
        return self.experiment['tasks']['checkpoint'].run(self.model, self.experiment, ckpt_info)

    def check_all_imports(self):
        import inspect
        for base_class in inspect.getmro(MainRunner):
            if 'check_imports' in base_class.__dict__.keys():
                base_class.check_imports(self)
