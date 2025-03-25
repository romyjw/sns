import hakowan as hkw
import lagrange
import math


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import os

SNSname = sys.argv[-1]


filepath = "../data/treefrog9919.obj"



base =  hkw.layer(filepath).material(
    "Principled",
    color="#2b8a44",
    roughness=0.2,
    ).rotate((1,0,0), 3.14 * -1/ 4).rotate((0,1,0), 3.14 * 1/ 4).rotate((0,0,1), 3.14 * 1/ 4)
    
    
wires = hkw.layer(filepath).mark("Curve").channel(size=0.002).material("Diffuse", "black").rotate((1,0,0), 3.14 * -1/ 4).rotate((0,1,0), 3.14 * 1/ 4).rotate((0,0,1), 3.14 * 1/ 4)
# Step 2: Adjust camera position.
config = hkw.config()
# Generate 4K rendering.
#config.film.width = 3840
#config.film.height = 2160

####### for spike ball
#config.sensor.location = [0, -3, 3]

########### for armadillo
#config.sensor.location = [3, 0, -3]
config.sensor.location = [0, 0, 3]

########### for igea
#config.sensor.location = [3, 0, 3]


############# for human
#config.sensor.location = [0, 0, 5]
# Step 3: Render the image.

            
hkw.render(base+wires, config, filename="../data/visualisation/treefrog9919/mesh.png")
