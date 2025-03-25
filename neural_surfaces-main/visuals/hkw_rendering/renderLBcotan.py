import hakowan as hkw
import lagrange
import math


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import os

SNSname = sys.argv[-1]


filepath = "../data/visualisation/LB_remesh/"+SNSname+"/"+SNSname+".ply"


	
base_dict = {}
n=4


for colouringname in ['scalarfield', 'cotanlb', 'mclb']:
    
    base_dict[colouringname] =  hkw.layer(filepath).material(
    "Principled",
    color=hkw.texture.ScalarField(
       data=colouringname, colormap='identity'
    ),
    roughness=0.6,
    )#.rotate((1,0,0), 3.14 * -1/ 4).rotate((0,1,0), 3.14 * 1/ 4).rotate((0,0,1), 3.14 * 1/ 4)
    
    
    #.rotate((1,0,0), 3.14*(-1/2)).rotate((0,1,0), 3.14*(-1/4))

    
    
    #
plain_base =  hkw.layer(filepath).material(
    "Principled",
    color="#FBCD50",
    roughness=0.2,
    )
    



wires = hkw.layer(filepath).mark("Curve").channel(size=0.002).material("Diffuse", "black")#.rotate((1,0,0), 3.14*(-1/2)).rotate((0,1,0), 3.14*(-1/4))
combo_view = (plain_base + wires)


# Step 2: Adjust camera position.
config = hkw.config()

config.sensor.location = [0, 1, 5]

config.film.width = 3840
config.film.height = 2160

#config.albedo_only=True

# Step 3: Render the image.

hkw.render(combo_view, config, filename="../data/visualisation/LB_remesh/"+SNSname+"/wires.png")


for colouringname, base in base_dict.items():
            
    hkw.render(base, config, filename="../data/visualisation/LB_remesh/"+SNSname+"/"+colouringname+"_orginalmesh.png")
