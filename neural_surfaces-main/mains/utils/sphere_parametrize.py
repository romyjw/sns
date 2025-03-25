
import numpy as np
import igl


def sphere_parametrize(V, F, method='manifold_optim', it=100, bnd_shape='circle'):#change


    if method == 'manifold_optim':
        return manifold_optim()
        
        
        
    elif method == 'arap':
        return arap(V, F, uv_init, bnd, bnd_uv, it)
    elif method == 'lscm':
        return lscm(V, F, uv_init, bnd, bnd_uv, it)



def manifold_optim():

    #sphere_param = igl.read_obj("../data/igea_final_embedding.obj")
    
    
    #sphere_param = igl.read_obj("../data/giraffe_final_embedding_THEIRS2.obj")
    
    #sphere_param = igl.read_obj("../data/giraffe_final_embedding.obj")
    #sphere_param = igl.read_obj("../analytic/sphere/sphere5.obj")
    
    #sphere_param = igl.read_obj("../analytic/sphere/sphere5.obj")
    
    sphere_param = igl.read_obj("../data/000temp_final_embedding.obj")

    print('I am sphere param: ', sphere_param)
    uva = sphere_param[0]
    
    return uva
    
    
     

def slim(V, F, uv_init, bnd, bnd_uv, it, bnd_constrain_weight):
    slim = igl.SLIM(V, F, uv_init, bnd, bnd_uv, igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET, bnd_constrain_weight)
    print(f'SLIM initial energy {slim.energy()}')
    count = 0
    slim.solve(it)
    while slim.energy() > 100.0:
        slim.solve(it)
        count += 1
        if count > 200:
            break
    # slim.solve(it)
    uva = slim.vertices()
    print(f'SLIM final energy {slim.energy()}')
    return uva


def arap(V, F, uv_init, bnd, bnd_uv, it):
    arap = igl.ARAP(V, F, 2, np.zeros(0))
    uva = arap.solve(np.zeros((0, 0)), uv_init)
    return uva


def lscm(V, F, uv_init, bnd, bnd_uv, it):
    _, uva = igl.lscm(V, F, bnd, bnd_uv)
    return uva


def map_cirlce_to_square(bnd_uv):
    u = bnd_uv[:,0].reshape(-1,1)
    v = bnd_uv[:,1].reshape(-1,1)
    u2 = np.power(u, 2)
    v2 = np.power(v, 2)
    sqrt_2 = np.sqrt(2)
    x = 0.5 * np.sqrt(2.0+2.0*sqrt_2*u+u2-v2) - 0.5* np.sqrt(2-2*sqrt_2*u+u2-v2)
    y = 0.5 * np.sqrt(2.0+2.0*sqrt_2*v-u2+v2) - 0.5* np.sqrt(2-2*sqrt_2*v-u2+v2)
    xy = np.concatenate([x,y], axis=1)
    return xy
