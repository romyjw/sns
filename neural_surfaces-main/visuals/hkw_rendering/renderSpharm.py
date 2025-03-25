import hakowan as hkw
import lagrange
import math


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import os


i = int(sys.argv[-1])

SNSname = 'icosphere5'
colouringname = 'eigfunc'



filepath = "../data/visualisation/eigenfunc/"+SNSname+"/"+SNSname+str(i)+".ply"

base_dict = {}

base =  hkw.layer(filepath).material(
"Principled",
color=hkw.texture.ScalarField(
   data=colouringname, colormap='identity'
),
roughness=0.3,
).rotate((0,1,0), 3.14 * (1/2)).rotate((1,0,0), 3.14 * (1/2)).rotate((0,1,0), 3.14 * (1/4)).rotate((1,0,0), 3.14 * (1/4)).rotate((0,1,0), 3.14 * (1/8))
    
    

# Step 2: Adjust camera position.
config = hkw.config()

config.sampler.sample_count = 512


#config.sensor = hkw.setup.sensor.Orthographic()
# Generate 4K rendering.
#config.film.width = 3840
#config.film.height = 2160

####### for spike ball
#config.sensor.location = [0, -3, 3]
########### for armadillo
config.sensor.location = [0, 0, 3]

########### for igea
#config.sensor.location = [3, 0, 3]

############# for human
# Step 3: Render the image.
hkw.render(base, config, filename="../data/visualisation/eigenfunc/"+SNSname+"/"+colouringname+str(i)+".png")
