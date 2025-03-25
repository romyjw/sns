import numpy as np
def write_curvature_ply_file(tm=None, H=None, K=None, Hmap=None, Kmap=None, filepath=None):

	mapped_meancurv = Hmap(H)
	mapped_gausscurv = Kmap(K)
	
	from plyfile import PlyData, PlyElement
	tm.export('../data/visualisation/temp.ply')
	p = PlyData.read('../data/visualisation/temp.ply')
	v = p.elements[0]
	f = p.elements[1]
	
	# Create the new vertex data with appropriate dtype
	a = np.empty(len(v.data), v.data.dtype.descr + [('meancurv', 'f8' ), ('gausscurv', 'f8'),('mapped_meancurv', 'f8' ), ('mapped_gausscurv', 'f8')])
	for name in v.data.dtype.fields:
	    a[name] = v[name]
	    
	a['meancurv'] = H
	a['gausscurv'] = K
	
	a['mapped_meancurv'] = mapped_meancurv
	a['mapped_gausscurv'] = mapped_gausscurv
	
	# Recreate the PlyElement instance
	v = PlyElement.describe(a, 'vertex')
	
	# Recreate the PlyData instance
	p = PlyData([v, f], text=True)
	p.write(filepath)




def write_custom_ply_file(tm=None, scalardict=None, filepath=None):

		
	from plyfile import PlyData, PlyElement
	tm.export('../data/visualisation/temp.ply')
	p = PlyData.read('../data/visualisation/temp.ply')
	v = p.elements[0]
	f = p.elements[1]
	
	
	
	something = [(scalarname, 'f8') for scalarname in scalardict]
	
	
	# Create the new vertex data with appropriate dtype
	a = np.empty(len(v.data), v.data.dtype.descr + something)
	
	
	
	for name in v.data.dtype.fields:
	    a[name] = v[name]
	   
	for scalarname, scalar in scalardict.items():
		a[scalarname] = scalar 	    

	
	# Recreate the PlyElement instance
	v = PlyElement.describe(a, 'vertex')
	
	# Recreate the PlyData instance
	p = PlyData([v, f], text=True)
	p.write(filepath)











def write_custom_colour_ply_file(tm=None, colouringdict=None, filepath=None):
	
	from plyfile import PlyData, PlyElement
	tm.export('../data/visualisation/temp.ply')
	p = PlyData.read('../data/visualisation/temp.ply')
	v = p.elements[0]
	f = p.elements[1]
	
	# Create the new vertex data with appropriate dtype
	extra_data_specs = [ (colouringname, 'f8', (3,)) for colouringname in colouringdict.keys() ]
	a = np.empty(len(v.data), v.data.dtype.descr + extra_data_specs)
	for name in v.data.dtype.fields:
	    a[name] = v[name]
	
	for colouringname,colouring in colouringdict.items():    
		a[colouringname] = colouring[:,:3]
	
	# Recreate the PlyElement instance
	v = PlyElement.describe(a, 'vertex')
	
	# Recreate the PlyData instance
	p = PlyData([v, f], text=True)
	p.write(filepath)