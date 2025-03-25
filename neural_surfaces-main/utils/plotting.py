from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np


def draw_histogram(data, range=None, bins=200, xlabel=None, plot_title=None, invisible=False):
	alpha=0.6
	color='g'
	edgecolor='black'
	
	if invisible==True:
		alpha=0.0
		edgecolor='white'
		
	count, bins, _ = plt.hist(data, range=range, bins=bins, density=True, alpha=alpha, color=color, edgecolor=edgecolor)
	if not xlabel is None:
		plt.xlabel(xlabel)
	if not plot_title is None:
		plt.title(plot_title)
	
	return count, bins, _

def draw_histogram_curve(data, range=None, bins=200, xlabel=None, label=None, plot_title=None, n=500, style='-', color=None):
	
	# Create the histogram
	count, bins, _ = draw_histogram(data, range=range, bins=bins, invisible=True)
	
	# Calculate the bin centers
	bin_centers = 0.5 * (bins[:-1] + bins[1:])
	
	# Create cubic interpolation model
	cubic_interpolation_model = interp1d(bin_centers, count, kind="cubic")
	
	# Create a fine grid for the x-axis
	x_fine = np.linspace(bin_centers[0], bin_centers[-1], n)
	y_fine = cubic_interpolation_model(x_fine)
	
	# Plot the cubic spline interpolation on top of the histogram
	plt.plot(x_fine, y_fine, ls=style, color=color, lw=1.5, label=label)
	
	# Add labels and legend
	if not xlabel is None:
		plt.xlabel(xlabel)
	if not plot_title is None:
		plt.title(plot_title)
		
	plt.ylabel('Density')
	plt.legend()