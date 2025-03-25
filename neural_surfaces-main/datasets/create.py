
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from .model import ModelDataset, ModelDatasetNoFaces

from .surface_map import SurfaceMapDataset, SurfaceMapSingularDataset
from .eigenfunc import EigenfuncDataset, EigenfuncOverfitDataset
from .geodesic import GeodesicDataset

def create(config, experiment):
    ### Create a dataset and dataloader for each element in the configuration

    loaders = {}
    for k, v in config.items():
        loaders[k] = create_loader(v)

    return loaders


def worker_init_function(id):
    return np.random.seed(torch.initial_seed() // 2**32 + id)

def create_loader(config):
    #print('CREATE LOADER CALLED')

    dataset = globals()[config['name']](config)

    sampler = None

    prefetch = config['batch_size'] if config['batch_size'] is not None else 10

    kwargs = {
        'sampler':sampler,
        'num_workers':config['num_workers'],
        'pin_memory':config['pin_memory'],
        'prefetch_factor':prefetch,
        'shuffle':config['shuffle'] if sampler is None else False,
        'worker_init_fn': worker_init_function
        #'worker_init_fn':lambda id: np.random.seed(torch.initial_seed() // 2**32 + id)
    }
    loader_class = DataLoader(dataset,config['batch_size'],**kwargs)
    

    return loader_class #(dataset, config['batch_size'], **kwargs)
