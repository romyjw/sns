import numpy as np

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-1*x) )
	
def inv_sigmoid(x):
	return -np.log((1-x)/x)
	
def mapping12(field,invert=False):
	if invert==False:
		return sigmoid(0.1*field)
	else:
		return 10*inv_sigmoid(x)

def mapping13(field):
	
	return sigmoid(0.1*np.sign(field)*(abs(field)**0.5))


def scaled_normals_cmap(x):
	ans = np.clip ( 0.02*np.abs(x) , 0.0, 1.0)
	return  ans

def scaled_normals_cmap2(x):
	ans = np.clip ( 0.02*x + 0.5 , 0.0, 1.0)
	return  ans

	
	
def linear5(x):
	return np.clip(x/20,-0.5,0.5) + 0.5
	
def linear6(x):
	return np.clip(0.0015*x,-0.5,0.5) + 0.5


def linear7(x):
	return np.clip(0.0004*x,-0.5,0.5) + 0.5
	


def positive_only_linear1(x):
	return np.clip(0.05*x, 0, 1)





def logmap(x):
	return linear(10*np.log(x))
	
def linear2(x):
	return np.clip(10*x,-0.5,0.5) + 0.5
	
def linear3(x):
	return np.clip(0.005*x,-0.5,0.5) + 0.5
	
	

	
	
def linear(x):
	return np.clip(x/20,-0.5,0.5) + 0.5
	
	
def quadratic(x):
	return np.clip(np.sign(x)*np.sqrt(abs(x))/20,-0.5,0.5) + 0.5
	
	
	
	
def linear8(x):
	return np.clip(x/10,-0.5,0.5) + 0.5	
	

def linear9(x):
	return np.clip(x/5,-0.5,0.5) + 0.5	
	
	
	
	
	
	
	
	
	
	