import hakowan as hkw
import lagrange
import math


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import os



name='spike25'

filepath = "../data/visualisation/MCF/"+name+".ply"
coarse_filepath = "../data/analytic/SPIKE/mesh5_nA.obj"

	
base_dict = {}


base =  hkw.layer(filepath).material(
    "Principled",
    color="#1e6fba",
    roughness=0.5,
    ).rotate((1,0,0), -3.14 / 4)
    
    
#glass = hkw.layer(coarse_filepath).material("ThinDielectric").rotate((1,0,0), -3.14 / 4)
    
#wires = hkw.layer(coarse_filepath).mark("Curve").channel(size=0.002).material("Diffuse", "black").rotate((1,0,0), -3.14 / 4)
#combo_view = (base + wires)


# Step 2: Adjust camera position.
config = hkw.config()
# Generate 4K rendering.
#config.film.width = 3840
#config.film.height = 2160


config.sensor.location = [0, 0, 3]



# Step 3: Render the image.
hkw.render(base, config, filename="../data/visualisation/MCF/"+name+".png")
#hkw.render(combo_view, config, filename="../data/visualisation/MCF/"+name+"+wires.png")
#hkw.render(base+glass, config, filename="../data/visualisation/MCF/"+name+"+with_glass.png")

############ render crossfield images ###################

