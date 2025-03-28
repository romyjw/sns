import numpy as np
import torch
from argparse import ArgumentParser
from pathlib import Path ### ???
import igl

from utils.mesh import *

import os

parser = ArgumentParser(description='convert mesh into Neural Surface Maps sample')
parser.add_argument('--data', type=str, help='path to file or folder containing mesh to convert', required=True)
parser.add_argument('--embedding', type=str, help='path to embedding mesh', required=True)
parser.add_argument('--pth_dir', type=str, help='intended path to pth file', required=False)

args = parser.parse_args()

meshfilename = args.data

pth_dir = args.pth_dir

input_path = Path(meshfilename)

embedding_meshfilename = args.embedding

embedding_path = Path(embedding_meshfilename)


shape_name = str(input_path.stem)

print('the shape name is ', shape_name)
points, faces, _, _, _ = readOBJ(input_path)
param, _, _, _, _ = readOBJ(embedding_path)

#points, faces, normals, V_idx, face_normals = clean_mesh(points, faces)
normals = igl.per_vertex_normals(points, faces)

points = np.array(points.tolist())

sample = {}
sample['param']      = torch.from_numpy(param).float() #just a placeholder, gets overwritten in 0automatic.
sample['points']         = torch.from_numpy(points).float()
sample['faces']          = torch.from_numpy(faces).long()
sample['normals']        = torch.from_numpy(normals).float()
sample['name'] = shape_name

## save parameterized mesh
output_file = '../data/'+shape_name + '_sphere.obj'

writeOBJ(output_file, sample['points'], sample['faces'], None, None)

print('I saved a sphere parametrisation in', output_file)



if pth_dir is None:

    ## save file as pth
    if not  os.path.isdir('../data/SNS/'+shape_name):
        os.mkdir('../data/SNS/'+shape_name)
    output_file = '../data/SNS/'+shape_name+ '/param' + '.pth'

else:
    ## save file as pth
    if not  os.path.isdir(pth_dir):
        os.mkdir(pth_dir)
    output_file = pth_dir + '/param' + '.pth'





torch.save(sample, output_file)
