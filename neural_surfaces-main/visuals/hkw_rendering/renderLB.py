import hakowan as hkw
import lagrange
import math


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import os

SNSname = sys.argv[-1]

'''
config = hkw.config()
filepath = "../data/"+SNSname+".obj"
thing = hkw.layer(filepath).material(
        "Principled",
        color=hkw.texture.ScalarField(
           data=normals, colormap='identity'
        ),
        roughness=0.6,
        )
    


hkw.render(thing, config, filename="../data/visualisation/LB_remesh/"+SNSname+"/thing_originalmesh.png")
'''





filepath = "../data/visualisation/LB_remesh/"+SNSname+"/"+SNSname+"_originalmesh.ply"


base_dict = {}

for colouringname in ['meshlb', 'meshlb_error']:
    
    try:
    
        base_dict[colouringname] =  hkw.layer(filepath).material(
        "Principled",
        color=hkw.texture.ScalarField(
           data=colouringname, colormap='identity'
        ),
        roughness=0.6,
        )
    except:
        pass

# Step 2: Adjust camera position.
config = hkw.config()

config.sensor.location = [3, 1, 5]## igea?


config.sensor.location = [0, -3, 0]


config.film.width = 3840
config.film.height = 2160



#config.albedo_only=True

# Step 3: Render the image.
for colouringname, base in base_dict.items():
            
    hkw.render(base, config, filename="../data/visualisation/LB_remesh/"+SNSname+"/"+colouringname+"_originalmesh.png")



plain_base =  hkw.layer(filepath).material(
    "Principled",
    color="#FBCD50",
    roughness=0.2,
    )


filepath = "../data/"+SNSname+".obj"


wires = hkw.layer(filepath).mark("Curve").channel(size=0.002).material("Diffuse", "black")
combo_view = (plain_base + wires)

hkw.render(combo_view, config, filename="../data/visualisation/LB_remesh/"+SNSname+"/wires_originalmesh.png")








filepath = "../data/visualisation/LB_remesh/"+SNSname+"/"+SNSname+"_densemesh.ply"


base_dict = {}

for colouringname in ['scalarfield', 'mclb',  'analylb', 'SNS_lb_error' ]:
#for colouringname in [ ]:

    base_dict[colouringname] =  hkw.layer(filepath).material(
    "Principled",
    color=hkw.texture.ScalarField(
       data=colouringname, colormap='identity'
    ),
    roughness=0.6,
    )

        

# Step 3: Render the image.
for colouringname, base in base_dict.items():
            
    hkw.render(base, config, filename="../data/visualisation/LB_remesh/"+SNSname+"/"+colouringname+"_densemesh.png")




plain_base =  hkw.layer(filepath).material(
    "Principled",
    color="#FBCD50",
    roughness=0.2,
    )
    




    
wires = hkw.layer(filepath).mark("Curve").channel(size=0.002).material("Diffuse", "black")
combo_view = (plain_base + wires)

hkw.render(combo_view, config, filename="../data/visualisation/LB_remesh/"+SNSname+"/wires_densemesh.png")





