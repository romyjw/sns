import hakowan as hkw
import lagrange
import math


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import os


print('yo')
SNSname = sys.argv[-1]

# Step 2: Adjust camera position.
config = hkw.config()

config.sensor.location = [0, -5, 5]


config.film.width = 3840
config.film.height = 2160


filepath = '../../../data/visualisation/'+SNSname+'/'+SNSname+'.ply'
for colouringname in ['colour_normals', 'colour_meancurv', 'colour_gausscurv' ]:


    base =  hkw.layer(filepath).material(
    "Principled",
    color=hkw.texture.ScalarField(
       data=colouringname, colormap='identity'
    ),
    roughness=0.6,
    ).rotate((0,0,1), 3.14 * 1/ 4)

            
    hkw.render(base, config, filename="../../../data/visualisation/"+SNSname+"/"+colouringname+".png")
    #hkw.render(base, config, filename="../../../data/visualisation/"+SNSname+"/"+colouringname+".pdf")
    
    
    

filepath = '../../../data/visualisation/'+SNSname+'/discrete_distortion.ply'

base = hkw.layer(filepath).material(
"Principled",
color=hkw.texture.ScalarField(
   data='colour_discrete_distortion', colormap='identity'
),
roughness=0.6,
).rotate((0,0,1), 3.14 * 1/ 4)


# Step 3: Render the image.
hkw.render(base, config, filename="../../../data/visualisation/"+SNSname+"/distortion.png")
#hkw.render(base, config, filename="../../../data/visualisation/"+SNSname+"/distortion.pdf")
    
filepath = '../../../data/visualisation/'+SNSname+'/discrete_distortion_sphere.ply'

base =  hkw.layer(filepath).material(
"Principled",
color=hkw.texture.ScalarField(
   data='colour_discrete_distortion', colormap='identity'
),
roughness=0.6,
).rotate((0,0,1), 3.14 * 1/ 4)


# Step 3: Render the image.

hkw.render(base, config, filename="../../../data/visualisation/"+SNSname+"/distortion_sphere.png")
    
    

