import matplotlib.pyplot as plt
import numpy as np

def create_colorbar(file_name, vmin, vmax, text_size, cmap='Reds'):
    """
    Create and save a horizontal colorbar.

    Parameters:
        file_name (str): Name of the output PNG file.
        vmin (float): Minimum value of the colorbar.
        vmax (float): Maximum value of the colorbar.
        text_size (int): Font size for the colorbar labels.
        cmap (str): Colormap to use.
    """
    # Create a dummy scalar mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    # Create the colorbar
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    #cbar.set_label(file_name, fontsize=text_size)
    cbar.ax.tick_params(labelsize=text_size)

    # Save the colorbar to a file
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Shared text size parameter
text_size = 20

# Create the three colorbars
create_colorbar('geometry_error_colourbar.png', 0, 0.005, text_size)
create_colorbar('H_and_K_colourbar.png', 0, 2, text_size)
create_colorbar('normals_mincurvdir_error_colourbar.png', 0, 30, text_size)

########## distortion figure ones ################
create_colorbar('distortion_colourbar.png', 0, 0.01, text_size, cmap='hot')
create_colorbar('meancurv_colorbar.png',   -2.5, 2.5, text_size, cmap='seismic')
