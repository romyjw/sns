import hakowan as hkw
import lagrange
import math


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Get the spectral colormap
cmap = plt.cm.get_cmap('Spectral')

# Get the hex values of the colormap
hex_values = [mcolors.to_hex(color) for color in cmap(np.linspace(0, 1, 256))]


# Customized color map.
#colormap = ["#a69c65", "#9A9A07", "#983A06", "#7C070A", "#160507", "#060103", "#000000"] #heat colours
colormap = hex_values





# Step 1: Create a base layer.
base = hkw.layer("../data/visualisation/test2.ply").material(
    "Principled",
    # We used isocontour texture to visualize the geodesic distance field both as color
    # and as isocurves.
    color=hkw.texture.Isocontour(
       data="malteaser",
        texture1=hkw.texture.ScalarField("malteaser", colormap=colormap),
        texture2="black",
        ratio=1.0,
        num_contours=100,
    ),
    
    #color=hkw.texture.ScalarField(
    #   data="malteaser", colormap=colormap
    #),


    roughness=0.5,
)



############ wireframe ############################
############################################
mesh = lagrange.io.load_mesh("../data/scan_018_nA.obj")
wires = hkw.layer().mark("Curve").channel(size=0.002).material("Diffuse", "black")

surface = (
    hkw.layer()
    .channel(normal="facet_normal")
    .material("Principled", "#FBCD50", roughness=0.2)
    .transform(hkw.transform.Compute(facet_normal="facet_normal"))
)

view = (surface+wires).data(mesh)
config = hkw.config()
config.sensor.location = [0, 1.2, 3]
hkw.render(view, config, filename="../data/visualisation/wireframe.png")
###########################################
###################################################




# Step 2: Adjust camera position.
config = hkw.config()
config.sensor.location = [0, 1.2, 3]

# Step 3: Render the image.
hkw.render(base, config, filename="../data/visualisation/bunny_heat.png")

# Step 4: Render the back side.
back_side = base.rotate(axis=[0, 1, 0], angle=math.pi)
hkw.render(back_side, config, filename="../data/visualisation/bunny_heat_back.png")