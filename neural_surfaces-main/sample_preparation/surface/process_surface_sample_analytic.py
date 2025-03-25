import numpy as np
import torch
from argparse import ArgumentParser
from pathlib import Path ### ???


#from utils import read_OBJ
#from utils import write_OBJ
import os
import numpy as np

parser = ArgumentParser(description='convert mesh into Neural Surface Maps sample')
parser.add_argument('--data', type=str, help='analytic shape name', required=True)
parser.add_argument('--num_points', type=int, help='number of points to sample', required=True)
args = parser.parse_args()

#meshfilename = args.data

#input_path = Path(meshfilename)
#shape_name = str(input_path.stem)


shape_name = args.data
n = args.num_points

formula_filepath = '../data/analytic/'+shape_name+'/formula.txt'
with open(formula_filepath) as formula_file:
	formula_string = formula_file.read()


# generate points on a sphere
P = np.random.randn(n, 3)
P /= np.linalg.norm(P, axis=1)[:, np.newaxis] 

U = np.arccos(P[:,2])###### arccos(z)
V = np.arctan2(P[:,1], P[:,0])

points = eval(formula_string) #evaluate analytic parametrisation
print('the shape name is ', shape_name)



#points, faces, _, _ = read_OBJ(input_file)
# remove ears vertices, and get normals (vtx normals and face normals)
#points, faces, normals, V_idx, face_normals = clean_mesh(points, faces) #where the action happens

points = np.array(points.tolist())

sample = {}
sample['param']      = torch.from_numpy(P).float() #just a placeholder, gets overwritten in 0automatic.
sample['points']         = torch.from_numpy(points).float()
sample['faces']          = None #no 'face sampling' when fitting directly to analytic
sample['normals']        = torch.zeros_like(sample['param']) #later we should include analytic normals
sample['name'] = shape_name

## save parameterized mesh
#output_file = '../data/'+shape_name + '_sphere.obj'

#write_OBJ(output_file, sample['points'], sample['faces'], sample['param'], None)

#print('I saved a sphere parametrisation in', output_file)

## save file as pth
if not os.path.isdir('../data/SNS/'+shape_name):
    os.mkdir('../data/SNS/'+shape_name)
output_file = '../data/SNS/'+shape_name+ '/param' + '.pth'

torch.save(sample, output_file)
