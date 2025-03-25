
import torch

import numpy as np

def prepare_sample(SNS_name, number_samples=1000):

	
    ### generate samples on an interval
    ### put into a dictionary format
    
    
    sample = {}
    
    even_samples = torch.linspace(0, 1, steps=number_samples).unsqueeze(1)
    print(even_samples.shape)	
    param = even_samples

    sample['param']      = param.detach()     
    sample['name']           = SNS_name

    return sample
    
