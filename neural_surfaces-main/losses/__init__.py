from .mse import MSELoss, MAELoss
from .ssd import SSDLoss
from .domain_surface_map import DomainSurfaceMapLoss

def create(config, experiment):
    loss = globals()[config['name']](**config['params'])

    return loss
