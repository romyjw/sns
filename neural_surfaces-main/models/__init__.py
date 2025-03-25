
import torchvision.models as models


from .mlp import MLP
from .mlp import ResidualMLP
from .neural_map import NeuralMap
from .neural_map import NeuralMapAngular
from .neural_map import NeuralMap3DAngular
from .neural_map import NeuralMapStereo
from .neural_map import NeuralMapSingular
from .neural_map import ParametrizationMap
from .surface_path_map import SurfacePathMap



def create(config, experiment):
    model = globals()[config['name']](config['structure'])

    return model
