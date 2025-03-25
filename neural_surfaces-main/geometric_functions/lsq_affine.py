import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def lsq_affine(Z, W): #Find complex a,b that minimise sum |az_i + b - w_i|^2
	n = Z.shape[0]
	
	a = ( n*(Z.conj()*W).sum() - W.sum()*(Z.conj().sum()) )  /  (n*(Z.abs()**2).sum() - Z.sum().abs()**2)
	
	b = W.sum()/n - a*Z.sum()/n
	
	return (a,b)

	
	
	
def test0():
	n=10
	
	# Z = torch.tensor([torch.tensor(1.j * 2.0 * m * torch.pi / n).exp() for m in range(n//3)])
	
	Z = torch.tensor([m*torch.tensor(1.j * 2.0 * m * torch.pi / n).exp() for m in range(n)])
	
	
	W = (3.0+12.j)*Z + 10.j + 6.0
	
	
	a,b = lsq_affine(Z,W)
	print(a,b)
	Z_ = a*Z + b
	print(Z_.shape)
	print(Z.shape)
	
	fig,ax = plt.subplots()
	plt.axis('equal')
	ax.scatter(Z.real,Z.imag, cmap = 'viridis')
	ax.scatter(W.real,W.imag, cmap = 'viridis')
	
	ax.scatter(Z_.real, Z_.imag, marker = 'x')
	plt.show()


# Define stereographic inverse map
def stereographic_inverse(z):
    x, y = z.real, z.imag
    denom = 1 + x**2 + y**2
    X = 2 * x / denom
    Y = 2 * y / denom
    Z = (-1 + x**2 + y**2) / denom
    return np.array([X, Y, Z])
    
    
    
def test1():
	n=10
	
	# Z = torch.tensor([torch.tensor(1.j * 2.0 * m * torch.pi / n).exp() for m in range(n//3)])
	
	Z = torch.tensor([m*torch.tensor(1.j * 2.0 * m * torch.pi / n).exp() for m in range(n)])
	
	
	W = ((3.0+12.j)/100)*Z 
	
	noise = np.sqrt(W.real**2 +W.imag**2) * np.random.randn(n)   
	
	W=W+noise
	
	
	a,b = lsq_affine(Z,W)
	print(a,b)
	Z_ = a*Z + b
	print(Z_.shape)
	print(Z.shape)
	
	fig,ax = plt.subplots()
	plt.axis('equal')
	ax.scatter(Z.real,Z.imag, cmap = 'viridis')
	ax.scatter(W.real,W.imag, cmap = 'viridis')
	
	ax.scatter(Z_.real, Z_.imag, marker = 'x')
	plt.show()
	
	
	# Map points to Riemann sphere
	Z_sphere = np.array([stereographic_inverse(z) for z in Z])
	W_sphere = np.array([stereographic_inverse(w) for w in W])
	Z__sphere = np.array([stereographic_inverse(z_) for z_ in Z_])
	
	# 3D plotting
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
	# Plot points on the Riemann sphere
	ax.scatter(Z_sphere[:, 0], Z_sphere[:, 1], Z_sphere[:, 2], cmap='viridis', label='Z')
	ax.scatter(W_sphere[:, 0], W_sphere[:, 1], W_sphere[:, 2], cmap='viridis', label='W')
	ax.scatter(Z__sphere[:, 0], Z__sphere[:, 1], Z__sphere[:, 2], marker='x', label='Z_')
	
	# Draw the sphere
	u = np.linspace(0, 2 * np.pi, 100)
	v = np.linspace(0, np.pi, 100)
	x = np.outer(np.cos(u), np.sin(v))
	y = np.outer(np.sin(u), np.sin(v))
	z = np.outer(np.ones(np.size(u)), np.cos(v))
	ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
	
	# Set plot properties
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	ax.legend()
	plt.axis('equal')
	plt.show()
	
	
	
test1()