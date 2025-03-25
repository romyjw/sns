import hakowan as hkw
import lagrange
import math


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import os

SNSname = sys.argv[-1]


render_crossfield = True
filepath = "../data/visualisation/"+SNSname+"/"+SNSname+".ply"


colouringdict = {
	'colour_normals' : True,
	'colour_meancurv' : False,	'colour_gausscurv' : False ,
	'colour_beltrami_H' : False,
	'colour_area_distortion' : False,
	'colour_beltrami_on_X' : False
	}
	
base_dict = {}

plain_base = hkw.layer(filepath).material(
    "Principled",
    color="#000050",
    roughness=0.3,
    )#.rotate((1,0,0), 3.14 * -1/ 2).rotate((0,1,0), 3.14 * 4/ 4)





for colouringname, value in colouringdict.items():
    base_dict[colouringname] =  hkw.layer(filepath).material(
    "Principled",
    color=hkw.texture.ScalarField(
       data=colouringname, colormap='identity'
    ),
    roughness=0.3,
    )#.rotate((1,0,0), 3.14 * -1/ 2).rotate((0,1,0), 3.14 * 4/ 4)
    #.rotate((1,0,0), 3.14 * -1/ 4).rotate((0,1,0), 3.14 * 1/ 4).rotate((0,0,1), 3.14 * 1/ 4) #froggo

# Step 2: Adjust camera position.
config = hkw.config()
# Generate 4K rendering.
config.film.width = 3840
config.film.height = 2160

####### for spike ball
config.sensor.location = [0, -3, 3]

########### for armadillo
#config.sensor.location = [3, 0, -3]
#config.sensor.location = [-3, 0, 3]
#config.sensor.location = [0, 1, 3]

########### for igea
#config.sensor.location = [3, 0, 3]

# Step 3: Render the image.


'''
for colouringname, base in base_dict.items():
    if colouringdict[colouringname]==True:
        
        if colouringname=='colour_normals':
            config.albedo_only = True
        else:
            config.albedo_only = False
            
        hkw.render(base, config, filename="../data/visualisation/"+SNSname+"/"+colouringname[7:]+".png")
        
        
config.albedo_only = False
hkw.render(plain_base, config, filename="../data/visualisation/"+SNSname+"/plain.png")
'''



############ render crossfield images ###################

if render_crossfield==True:
    
    base = hkw.layer(filepath).material(
    "Principled",
    "#FBCD50",
    roughness=0.3,
    )#.rotate((1,0,0), 3.14 * -1/ 2).rotate((0,1,0), 3.14 * 4/ 4)
    
    min_dir = hkw.layer("../data/visualisation/"+SNSname+"/crossfield_min_dir.obj").material(
        "Principled", "#FF0000",
    
        roughness=0.3,
    )#.rotate((1,0,0), 3.14 * -1/ 2).rotate((0,1,0), 3.14 * 4/ 4)
    
    max_dir = hkw.layer("../data/visualisation/"+SNSname+"/crossfield_max_dir.obj").material(
        "Principled", "#0000FF",
    
        roughness=0.3,
    )#.rotate((1,0,0), 3.14 * -1/ 2).rotate((0,1,0), 3.14 * 4/ 4)
    
    hkw.render(base + min_dir, config, filename="../data/visualisation/"+SNSname+"/min_dir.png")
    hkw.render(base + max_dir, config, filename="../data/visualisation/"+SNSname+"/max_dir.png")
    hkw.render(base + min_dir+max_dir, config, filename="../data/visualisation/"+SNSname+"/both_dir.png")
