
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Identity

from .utils.create_layers import create_sequential_linear_layer
from .utils.create_layers import get_init_fun

import torch





class MLP(Module):
    ## simple MLP network

    def __init__(self, config):
        super().__init__()

        in_size    = config['input_size'] # input size of MLP
        out_size   = config['output_size'] # output size of MLP
        layers     = config['layers'] # hidden layers size
        act_name   = config.get('act', 'Softplus') # activation for the MLP
        norm_name  = config.get('norm', None) # normalization layers (if any)
        drop_prob  = config.get('drop', 0.0) # dropout probability (if any)
        bias       = config.get('bias', True) # bias
        act_params = config.get('act_params', {}) # parameters for the activation function
        try:
            init_path = config['init_path']
        except:
            init_path = 'None'

        layers = [ in_size ] + layers + [ out_size ]

        self.mlp = create_sequential_linear_layer(layers, act_name, norm_name, drop_prob, bias, last_act=False, act_params=act_params)
        
        if not (init_path == 'None'):
            init_fun = get_init_fun(config['init'])
            self.mlp.apply(init_fun)
        else:
            pass


    def forward(self, x):
        return self.mlp(x)



class ResidualMLPBlock(Module):

    def __init__(self, in_features, act_fun, norm_layer, drop_prob, bias, act_params, out_features=None):
        super().__init__()

        layers = [in_features]*3
        if out_features is not None:
            layers[-1] = out_features

        layer = create_sequential_linear_layer(layers, act_fun, norm_layer, drop_prob, bias, last_act=True, act_params=act_params)

        self.shortcut = Identity()
        if in_features != out_features and out_features is not None:
            self.shortcut = create_sequential_linear_layer([in_features, out_features], act_fun, norm_layer, drop_prob, bias, last_act=False)

        self.residual = Sequential(*layer[:-1])
        self.post_act = layer[-1]


    def forward(self, x):
        x   = self.shortcut(x)
        res = self.residual(x)
        out = self.post_act(res + x)
        return out



class ResidualMLP(Module):

    def __init__(self, config):
        super().__init__()

        print(config.keys())
        in_size    = config['input_size'] # input size of MLP
        out_size   = config['output_size'] # output size of MLP
        layers     = config['layers'] # hidden layers size
        act_name   = config.get('act', 'Softplus') # activation for the MLP
        norm_name  = config.get('norm', None) # normalization layers (if any)
        drop_prob  = config.get('drop', 0.0) # dropout probability (if any)
        bias       = config.get('bias', True) # bias
        act_params = config.get('act_params', {}) # parameters for the activation function
        try:
            init_path = config['init_path']
        except:
            init_path = 'None'

        modules = []

        ## create first layer
        layer = create_sequential_linear_layer([in_size, layers[0]], act_name, norm_name, drop_prob, bias, last_act=True, act_params=act_params)
        modules.extend([el for el in layer])

        ## create residual blocks
        for layer in layers:
            block = ResidualMLPBlock(layer, act_name, norm_name, drop_prob, bias, act_params)
            modules.append(block)

        ## create last layer
        layer = create_sequential_linear_layer([layers[-1], out_size], act_name, norm_name, drop_prob, bias, last_act=False)
        modules.extend([el for el in layer])

        ## assemble into a single sequential
        self.mlp = Sequential(*modules)
        print(self.mlp)
        ## initialize weights
        
        #init_fun = get_init_fun(config['init'])
        #self.mlp.apply(init_fun)
        
        self.init_map_weights()    #####delete above two lines and replace with this line if you want to init with identity.
        print('init_path is',init_path)        
        ## set an init_path in the json if you would like to use an initial set of weights (e.g. for finetuning)    
        if not (init_path=='None'):          
            init_weights = torch.load(init_path, map_location=torch.device('cpu'))
            print('init weights keys',init_weights.keys())
            self.load_state_dict(init_weights)
            print('loaded init weights from file')
    def init_map_weights(self):
        ######## use this in combination with alteration to forward(self,x) to initialise with identity.

        print('Res MLP init map weights called')  
        # init with identity by adding point at the end                                                                                                                                    
        with torch.no_grad():                                                                                                                                                               
            if hasattr(self.mlp[-1], 'weight'): 
                print('yh 1')                                                                                                                               
                self.mlp[-1].weight.fill_(0.0001)      
                #if hasattr(self.neural_map.mlp[-1], 'bias'):                                                                                                                             
                #    self.neural_map.mlp[-1].bias.fill_(0.0)                                                                                                                                    
            else:  
                print('yh 2')
                                                                                                                                                           
                for i in range(len(self.mlp)):                                                                                                                                  
                    self.mlp[i].residual[2].weight *= 0.01                                                                                                                       
                    #self.neural_map.mlp[i].residual[2].bias *= 0.01  


    def forward(self, x):
        #x = torch.tensor([1,1,5])*(x + self.mlp(x)) ###### to initialise with identity!!!
        #x = torch.tensor([1,1,5]).to(x.device)*( self.mlp(x))
        x = self.mlp(x) ###### usual version
            
        return x

