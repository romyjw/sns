import hakowan as hkw
import lagrange
import math


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import os

SNSname = sys.argv[-1]


filepath = "../data/visualisation/heat/"+SNSname+"/heatflow.ply"




	
base_dict = {}
n=29

for i in range(n):
    colouringname = 'frame_'+str(i)
    base_dict[colouringname] =  hkw.layer(filepath).material(
    "Principled",
    color=hkw.texture.ScalarField(
       data=colouringname, colormap='identity'
    ),
    roughness=0.6,
    ).rotate((0,0,1), 3.14 * 2/ 4).rotate((0,1,0), 3.14 * 1/ 4)
    
    
    #.rotate((1,0,0), 3.14 * 3/ 4).rotate((0,0,1), 3.14 * 4/ 4).rotate((1,0,0), 3.14 * 1/ 8).rotate((0,1,0), 3.14 * 1/ 8) #for max10606
    
    
    
    #.scale(0.7).rotate((1,0,0), 3.14 * (-1/16)).rotate((0,1,0), 3.14 * (3/4))#scan_003
    #.rotate((0,1,0), 3.14 * (-1/4)) 
    
    #.scale(0.7).rotate((1,0,0), 3.14 * (-1/16)).rotate((0,1,0), 3.14 * (5/4))#scan_003
    #.rotate((0,1,0), 3.14 * (-1/8)) 
    #no rotation for scan_018
    
    #.rotate((1,0,0), 3.14 * 3/ 4).rotate((0,0,1), 3.14 * 4/ 4).rotate((1,0,0), 3.14 * 1/ 8).rotate((0,1,0), 3.14 * 1/ 4) #for max10606

# Step 2: Adjust camera position.
config = hkw.config()
# Generate 4K rendering.
#config.film.width = 3840
#config.film.height = 2160

####### for spike ball
#config.sensor.location = [0, -3, 3]

########### for armadillo
#config.sensor.location = [3, 0, -3]
config.sensor.location = [0, 2, 3]

########### for igea
#config.sensor.location = [3, 0, 3]


############# for human
#config.sensor.location = [0, 0, 5]
# Step 3: Render the image.
for colouringname, base in base_dict.items():
            
    hkw.render(base, config, filename="../data/visualisation/heat/"+SNSname+"/"+colouringname+".png")
