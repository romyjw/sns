from scipy import sparse
import trimesh
import numpy as np

def uniform_laplace(n,nbrs):#takes number of vertices and vertex neighbours (onerings) as input
    
    
    L = sparse.lil_matrix((n,n))
    
    for j in range(n):
        L[j,nbrs[j]] = 1/len(nbrs[j])
        L[j,j] = -1
    L = sparse.csr_matrix(L)
    
    return L
    


def laplace_beltrami_cotan_MC(v,onerings,edge_vertices, ignore_boundary=True):#takes vertices and mesh as input
    nbrs = onerings
    
    M = sparse.lil_matrix((v.shape[0],v.shape[0]))
    Minv = sparse.lil_matrix((v.shape[0],v.shape[0]))
    C = sparse.lil_matrix((v.shape[0],v.shape[0]))
    
    for j in range(v.shape[0]):

        my_nbrs = nbrs[j].copy()
        val = len(my_nbrs)
        
        if edge_vertices[j]==1 and ignore_boundary==True:#leave row as all zeros if edge vertex.
            pass
        elif edge_vertices[j]==1:#ignore boundary is False
            magnitudes = np.array([np.linalg.norm(v[j,:] - v[my_nbrs[0],:]) , np.linalg.norm(v[j,:] -     v[my_nbrs[-1],:])])
            #print(magnitudes)
            #print(edge_vertices[my_nbrs[0]], edge_vertices[my_nbrs[-1]], j)
            C[j,my_nbrs[0]] = 1/magnitudes[0]
            C[j,my_nbrs[-1]] = 1/magnitudes[1]
            C[j,j] = - (1/magnitudes[0] + 1/magnitudes[1])
            M[j,j] = np.sum(magnitudes)/2
            Minv[j,j] = 2/np.sum(magnitudes)
            
        else:
        
            #####################work out cotan terms#####################
            
            dot_products_row1 = np.array([np.sum((v[j,:] - v[my_nbrs[(i+1)%val],:])*
                                     (v[my_nbrs[i],:] - v[my_nbrs[(i+1)%val],:]))
                                 for i in range(val)])#dot product gives (ab)cosC for each face
            
            magnitudes_row1 = np.array([#product of magnitudes (i.e the ab for each face)
                (np.linalg.norm(v[j,:] - v[my_nbrs[(i+1)%val],:])*
                         np.linalg.norm(v[my_nbrs[i],:] - v[my_nbrs[(i+1)%val],:]))
                                    for i in range(val)])
            
            
            dot_products_row2 = np.array([np.sum((v[j,:] - v[my_nbrs[(i-1)%val],:])*
                                    (v[my_nbrs[i],:] - v[my_nbrs[(i-1)%val],:]))
                                        for i in range(val)])#dot product gives (ab)cosC for each face (for the other set of angles)
            
            
            magnitudes_row2 = np.array([#magnitudes of dot products (i.e the ab for each face) (for the other set of edge pairs)
                (np.linalg.norm(v[j,:] - v[my_nbrs[(i-1)%val],:])*
                        np.linalg.norm(v[my_nbrs[i],:] - v[my_nbrs[(i-1)%val],:]))
                                    for i in range(val)])
            
            cosines = np.vstack([dot_products_row1/magnitudes_row1,
                                 dot_products_row2/magnitudes_row2])
            #print(cosines.shape)
            
            
            cosines = np.clip(cosines, -1,1)
            
            sines = np.clip((1 - cosines**2)**0.5, 0, 1)
            
            
            cotans = cosines/sines
            
            #####################work out area normalisation#####################
            
            
            dot_products = np.array([np.sum((v[j,:] - v[my_nbrs[i],:])*
                                            (v[j,:] - v[my_nbrs[(i+1)%val],:]))
                                for i in range(val)])#dot product gives (ab)cosC for each face
            
            magnitudes = np.array([#magnitudes of dot products (i.e the ab for each face)
                (np.linalg.norm(v[j,:] - v[my_nbrs[i],:])*
                 np.linalg.norm(v[j,:] - v[my_nbrs[(i+1)%val],:])) 
                                for i in range(val)])
            
            cosines = np.clip(dot_products/magnitudes, -1,1)
            
            sines = np.clip((1 - cosines**2)**0.5, 0, 1)
            
            total_area = np.sum(0.5*magnitudes*sines)#because area of triangle is (1/2)absinC
            
            
            #total_area = 1.0#temporary

            C[j,nbrs[j]] = np.sum(cotans, axis = 0)
            C[j,j] = -1*np.sum(cotans)# -sum of all cot_alpha_ij + cot_beta_ij
                                                         #for each i
            
            M[j,j] = 2 * total_area/3
            Minv[j,j] = 1/(2*total_area/3)
       
    M = sparse.csr_matrix(M)
    Minv = sparse.csr_matrix(Minv)
    C = sparse.csr_matrix(C)
    L = Minv@C
    
    return L,M,Minv,C

def sort_onerings(face_nbrs,faces):
    sorted_nbrs = []
    edge_vertices = np.zeros(face_nbrs.shape[0])
    print('Sorting vertex neighbourhoods...')
    for i in range(face_nbrs.shape[0]):
        if i%1000==0:
            print(i,'of',face_nbrs.shape[0])
        my_face_nbrs = [face_nbrs[i,j] for j in range(face_nbrs.shape[1]) ]
        
        #not_my_face_nbrs = [face_nbrs[j,i] for j in range(face_nbrs.shape[1]) ]
        
        thing=[faces[my_face_nbrs[i]] for i in range(len(my_face_nbrs))]
        
        #print('i=',i,thing)
        
        
        if -1 in my_face_nbrs:
            my_face_nbrs.remove(-1)
        my_sorted_nbrs = []
        
        curF = faces[my_face_nbrs[0]]#pick a first face
        curV = min([v for v in curF if v!=i])#pick a first vertex
        nxtV = max([v for v in curF if v!=i])#make other vertex next vertex
        #for later:
        F0 = curF.copy()
        V0 = curV.copy()
        V1 = nxtV.copy()
        
        my_sorted_nbrs.append(curV)
        options = [face for face in my_face_nbrs if
                   (nxtV in faces[face] and not curV in faces[face]) and i in faces[face]]        
        
        while len(options)>0 and nxtV!=V0:
            curF = faces[options[0]]
            curV = nxtV.copy()
            nxtV = min([v for v in curF if (v!=i and v!=curV)])#make other vertex next vertex
            
            my_sorted_nbrs.append(curV)#put in current vertex
            
            options = [face for face in my_face_nbrs if#list of face indices
                   (nxtV in faces[face] and not curV in faces[face])] 
            #all the faces in the ring that contain nxtV (but not the same face as before) 
        
        if nxtV != V0:#i.e. it's not a loop
            my_sorted_nbrs.append(nxtV)
            edge_vertices[i]=1#flag that it's not a loop
            options = [face for face in my_face_nbrs if
                   (V0 in faces[face] and not V1 in faces[face]) and i in faces[face]] 
            
            #print('V1 is',V1)
            #print('V0 is',V0)
            nxtV = V0.copy()
            
            while len(options)>0:
                curF = faces[options[0]]
                curV = nxtV.copy()
                nxtV = min([v for v in curF if (v!=i and v!=curV)])#make other vertex next vertex
                #print('nxt v',nxtV)
                
                if curV!=V0:
                    #edge_vertices[curV]=2
                    my_sorted_nbrs = [curV] + my_sorted_nbrs#put in current vertex

                options = [face for face in my_face_nbrs if
                   (nxtV in faces[face] and not curV in faces[face])] 
                #print(faces[options], 'options for index',i)
                
                #all the faces in the ring that contain nxtV (but not the same face as before) 
        	
            my_sorted_nbrs = [nxtV] + my_sorted_nbrs#TRY
            #print(my_sorted_nbrs,'index',i)
        sorted_nbrs.append(my_sorted_nbrs)
    print('Done.')
    return sorted_nbrs,edge_vertices


def L_uniform(tm):

	vertices = tm.vertices
	faces = tm.faces
	n = vertices.shape[0]
	
	face_nbrs = tm.vertex_faces
	nbrs,edge_vertices = sort_onerings(face_nbrs,faces)
	L = uniform_laplace(n, nbrs)
	return L

def L_cotan(tm):

	vertices = tm.vertices
	faces = tm.faces
	n = vertices.shape[0]
	
	face_nbrs = tm.vertex_faces
	nbrs,edge_vertices = sort_onerings(face_nbrs,faces)
	
	L,M,Minv,C = laplace_beltrami_cotan_MC(vertices,nbrs,edge_vertices)
	
	return L,M,Minv,C

