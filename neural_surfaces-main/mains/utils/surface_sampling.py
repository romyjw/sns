
import torch
from torch.nn import functional as F



def sample_surface(num_samples, points, faces, params_to_sample=[], weights=None, method='pytorch3d'):
    #print('SURFACE SAMPLING CALLED')

    if method == 'pytorch3d':
        return sample_surface_pytorch3d(num_samples, points, faces, params_to_sample, weights)
    elif method == '3Dgaussian':
        return gaussian_sphere_sampling(num_samples)
    else:
        return sample_surface_trimesh(num_samples, points, faces, params_to_sample)



def gaussian_sphere_sampling(num_samples):
    samples = torch.random.randn(3,num_samples)
    samples = (samples/torch.sqrt(torch.einsum('ij,ij->j',samples,samples))).T
    #now num_samples x 3

    return samples
    
    
def sample_surface_pytorch3d(num_samples, points, faces, params_to_sample, weights=None):

    tris = points[faces]
    #if points.size(1) < 3:
    #    tris = F.pad(tris, (0, 1))
    vec_cross = torch.cross(tris[:, 0] - tris[:, 2],
                            tris[:, 1] - tris[:, 2], dim=-1)

    if weights is None:
        # weights = torch.ones(faces.size(0))
        weights = vec_cross.squeeze(-1).pow(2).sum(-1) # face area
    faces_idx = torch.multinomial(weights, num_samples, replacement=True)#choose some faces

    bar_coo = rand_barycentric_coords(num_samples) # w0, w1, w2 choose barycentric coords in each face

    # build up points from samples
    # P = w0 * A + w1 * B + w2 * C
    P = sample_triangles(tris, faces_idx, bar_coo)
    if points.size(1) < 3:#Romy: this makes it work for either 2d or 3d domain
        P = P[:, :2]

    normals = F.normalize(vec_cross, dim=-1)[faces_idx]

    # get vertices
    if len(params_to_sample) > 0:

        params_samples = []
        for param in params_to_sample:
            p = sample_triangles(param[faces], faces_idx, bar_coo)
            params_samples.append(p)
            
            
            
            
        #print(P.shape)
        
      
        #import matplotlib.pyplot as plt
        #plt.scatter(P[:,0],P[:,1])

        #plt.show()
        
        #print(len(params_samples))
        #print(params_samples[0].shape)
        
        #P2 = params_samples[0]
        
        #import matplotlib.pyplot as plt
        
        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        
        #ax.scatter(P2[:,0],P2[:,1],P2[:,2], marker='o', color='r')

        #plt.show()
        
        #print(params_samples[1].shape)
        #print(params_samples[2].shape)
        #print(params_samples[3].shape)
        
        
        return P, normals, params_samples
    
    

    return P, normals


def sample_triangles(tris, selected_faces, bar_coo):
    A = tris[selected_faces, 0]#1st coord
    B = tris[selected_faces, 1]#2nd coord
    C = tris[selected_faces, 2]#3rd coord

    # build up points from samples
    samples = bar_coo[0] * A + bar_coo[1] * B + bar_coo[2] * C
    
    #print(samples.shape)
    
    #samples = (samples.T/torch.sqrt(torch.einsum('ij,ij->i',samples,samples))).T      #####very bad!!!! silly.
    


    return samples


def rand_barycentric_coords(num_points):

    uv = torch.rand(num_points, 2)
    u, v = uv[:, 0:1], uv[:, 1:]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    # if torch.isnan(w0).any() or torch.isnan(w1).any() or torch.isnan(w2).any():
    #     return rand_barycentric_coords(num_points)

    return w0, w1, w2


def sample_surface_trimesh(num_samples, points, faces, param):

    faces_idx = torch.multinomial(torch.ones(faces.size(0)), num_samples, replacement=True)

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = torch.rand((num_samples, 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(dim=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = random_lengths.abs()

    points_samples = compute_samples(points, faces, faces_idx, random_lengths)

    if param is not None:
        params_samples = compute_samples(param, faces, faces_idx, random_lengths)

        return points_samples, params_samples

    return points_samples


def compute_samples(points, faces, faces_idx, random_lengths):
    triangles    = points[faces]
    tri_origins  = triangles[:, 0]
    tri_vectors  = triangles[:, 1:].clone()
    tri_vectors -= tri_origins.unsqueeze(1)

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[faces_idx]
    tri_vectors = tri_vectors[faces_idx]

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(dim=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    points = sample_vector + tri_origins

    return points
