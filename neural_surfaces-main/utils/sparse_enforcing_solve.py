from scipy import sparse
def sparse_enforcing_solve(A, B, sparsity): ### sparisty is a list of indices (i,j) such that X[i,j] is not enforced to be zero
	m,n = A.shape

	

	################################
	phi = dict() ## put in k, get out (i,j)
	phi_inv = dict()
	
	k=0
	for location in sparsity:
		phi[location] = k
		phi_inv[k] = location
		k+=1
	######################################
	
	
	q = len(sparsity) #number of nonzero entries
	print('m,n,q',m,n,q)
	
	
	####################################
	
	######################################
	
	#for k in range(q): ### construct the Adash matrix
	#	i,j = phi_inv[k]
	#	Adash[n*i+j,k] = A[] # set entry in position k in the n*i + jth row
	
	Adash = sparse.lil_matrix((m*n, q)) # mn x q sparse matrix
	for (j,s) in phi.keys():
		k = phi[(j,s)]
		Adash[j::n, k] = A[:,s] # Adash[n*i + j, k] = A[i, s] #the ith function evaluated at vertex s
				
	Bdash = B.reshape((m*n,1)) #flatten the B matrix
		
	
	Xdash, istop, itn, normr = sparse.linalg.lsqr(Adash, Bdash)[:4] ### do the sparse solve
	
	X = sparse.lil_matrix((n,n))
	
	for k in range(q): ### put the values of Xdash into a n x n matrix X
		i,j = phi_inv[k]
		X[i,j] = Xdash[k]
	
	
	return X #return dense matrix
		
	