import hakowan as hkw
import lagrange
import math


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import os

SNSname = 'treefrog9919'



shape_filepath = "../data/visualisation/"+SNSname+"/"+SNSname+"scalar.ply"
sphere_filepath = "../data/visualisation/"+SNSname+"/"+SNSname+"scalar_sphere.ply"




shape =  hkw.layer(shape_filepath).material(
    "Principled",
    color=hkw.texture.ScalarField(
       data='scalar', colormap='identity'
    ),
    roughness=0.3,
    ).rotate((1,0,0), 3.14 * -1/ 4).rotate((0,1,0), 3.14 * 1/ 4).rotate((0,0,1), 3.14 * 1/ 4)

sphere =  hkw.layer(sphere_filepath).material(
    "Principled",
    color=hkw.texture.ScalarField(
       data='scalar', colormap='identity'
    ),
    roughness=0.3,
    ).rotate((0,1,0), 3.14 * 1/ 4)
    
    


# Step 2: Adjust camera position.
config = hkw.config()
# Generate 4K rendering.
#config.film.width = 3840
#config.film.height = 2160

####### for spike ball
#config.sensor.location = [0, -3, 3]

########### for armadillo
#config.sensor.location = [3, 0, -3]
config.sensor.location = [0, 1, 3]

########### for igea
#config.sensor.location = [3, 0, 3]

# Step 3: Render the image.
#hkw.render(shape, config, filename="../data/visualisation/"+SNSname+"/scalar.png")

hkw.render(sphere, config, filename="../data/visualisation/"+SNSname+"/scalar_sphere.png")
