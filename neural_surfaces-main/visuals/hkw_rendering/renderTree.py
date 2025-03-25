import hakowan as hkw
import lagrange
import math


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import os





# Step 2: Adjust camera position.
config = hkw.config()

config.sensor.location = [0, -7, 5]


config.film.width = 3840
config.film.height = 2160



for meshname in ['SMALLTREE4', 'SMALLTREE5', 'SMALLTREE6']:

    filepath = "../data/"+meshname+".obj"
    
    plain_base =  hkw.layer(filepath).material(
        "Principled",
        color="#FBCD50",
        roughness=0.2,
        )


    wires = hkw.layer(filepath).mark("Curve").channel(size=0.001).material("Diffuse", "black")
    combo_view = (plain_base + wires)
    
    hkw.render(combo_view, config, filename="../data/visualisation/" + meshname + ".png")
    #hkw.render(combo_view, config, filename="../data/visualisation/" + meshname + ".pdf")
    
    

