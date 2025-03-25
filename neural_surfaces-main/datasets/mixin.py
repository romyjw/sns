
import numpy as np
import pickle
import torch


class DatasetMixin:

    def __len__(self):
        return self.num_epochs if not self.validation else 1

    def read_sample(self, path):
        if path[-3:] == 'pkl':
            sample = self.read_pickle_sample(path)
        elif path[-3:] == 'pth':
            sample = self.read_torch_sample(path)
        else:
            sample = self.read_numpy_sample(path)
        return sample

    def read_pickle_sample(self, path):
        return pickle.load(open(path, 'rb'))

    def read_torch_sample(self, path):
        return torch.load(path, map_location='cpu')

    def read_numpy_sample(self, path):
        return torch.from_numpy(np.load(path))

   
    # split a set of points into 'num_blocks' batches
    def split_to_blocks(self, size, num_blocks):
        idxs = torch.randperm(size)
        block_size = int(float(idxs.size(0)) / float(num_blocks))
        blocks = []
        for i in range(num_blocks):
            blocks.append(idxs[block_size * i : block_size * (i + 1)])

        return blocks

    def compute_lands_rotation(self, lands_source, lands_target):
        ### compute rotation matrix

        with torch.no_grad(): # not sure if this is necessary
            # R * X^T = Y
            center_lands_source = lands_source - lands_source.mean(dim=0)
            center_lands_target = lands_target - lands_target.mean(dim=0)
            H = center_lands_source.transpose(0,1).matmul(center_lands_target)
            u, e, v = torch.svd(H)
            R = v.matmul(u.transpose(0,1)).detach()

            # check rotation is not a reflection
            if R.det() < 0.0:
                v[:, -1] *= -1
                R = v.matmul(u.transpose(0,1)).detach()

        #t = lands_target.mean(dim=0) - lands_source.mean(dim=0).matmul(R.t())
        #print('rotation matrix:',R)
        return R
    
    
    def compute_lands_inversion(self, lands_source, lands_target):
       
        
        from inversion import compute_inversion_sig, invert_sphere1, invert_sphere2
        
        #S_correspondences = torch.cat([lands_source[:2,:] , lands_target[:2,:] ],0) ##### Check if correct way around.
        
        #temp_lands = torch.tensor([[-1.0/(2.0**0.5), 1.0/(2.0**0.5), 0.0],[-1.0/(2.0**0.5), -1.0/(2.0**0.5), 0.0]])
        
        #inversion_sig1 = compute_inversion_sig(temp_lands, default=False) ####manually choosing better-separated ldmks!!
        #inversion_sig2 = compute_inversion_sig(temp_lands, default=False)
        
        inversion_sig1 = compute_inversion_sig(lands_source[:,:]) #### now you can pass the indices of the chosen 'special landmarks'. 
        inversion_sig2 = compute_inversion_sig(lands_target[:,:])

        
        return (inversion_sig1, inversion_sig2)
        
        
        
    def compute_lands_rotate2pole(self, lands_source, lands_target):

        from rotations import compute_rotation_sig, rotate2pole1, rotate2pole2
               
        rotation_sig1 = compute_rotation_sig(lands_source[:,:] + 0.00001) #### now you can pass the indices of the chosen 'special landmarks'. 
        rotation_sig2 = compute_rotation_sig(lands_target[:,:] + 0.000001)
        
        
        ###### going to implement a 2D rotation as well. #######
        rotation2D = None ### compute_rotation2D(lands_source, lands_target, rotation_sig1, rotation_sig2)

        return (rotation_sig1, rotation_sig2)

     
    def compute_lands_mobius_triplet_inversion(self, lands_source, lands_target):
       
        from inversion import compute_inversion_sig, invert_sphere1, invert_sphere1, compute_full_mobius_sig, compute_half_mobius_sig
               
        mobius_sig = compute_full_mobius_sig(lands_source[:,:], lands_target[:,:])

        return mobius_sig
        
    
    
    def compute_lands_lsq_mobius(self, lands_source, lands_target):
        
        from mobius_triplet import stereographic, stereographic_inv
        from inversion import compute_inversion_sig, invert_sphere1, invert_sphere1, compute_full_mobius_sig, compute_half_mobius_sig
        from rotations import compute_rotation_sig, rotate2pole1, rotate2pole2
        from lsq_affine import lsq_affine
        
        rotation_sig1 = compute_rotation_sig( lands_source[:,:] )
        rotation_sig2 = compute_rotation_sig( lands_target[:,:] )
        
        pre_affine_lands = stereographic ( rotate2pole1(lands_source[1:,:], rotation_sig1) )
        post_affine_lands = stereographic (   rotate2pole1(lands_target[1:,:], rotation_sig2)   )
        
        a,b = lsq_affine (pre_affine_lands, post_affine_lands)
        print('abs-in:',pre_affine_lands.abs())
        print('abs-out:', post_affine_lands.abs())
        
                                           
        output_lands = rotate2pole2 (      stereographic_inv ( a*pre_affine_lands + b )          , rotation_sig2)
        output_lands_ext = torch.zeros_like(lands_source)
        output_lands_ext[0,:] = rotate2pole2(torch.tensor([[0.0, 0.0, 1.0]]), rotation_sig2)
        output_lands_ext[1:,:] = output_lands        
    
        mobius_sig = compute_full_mobius_sig(lands_source[:,:], output_lands_ext[:,:])

        return mobius_sig #the signature format is exactly the same as for triplet mobius - we just construct the triplets before training.
    
       
    def compute_lands_mobius(self, lands_source, lands_target):
        ### compute mobius transform
        
        #mobius_sig_src = lands_source[:3,:]
        #mobius_sig_tgt = lands_target[:3,:]
        #mobius_sig = (mobius_sig_src, mobius_sig_tgt)
        
        from mobius_triplet import find_mobius_from_3corresp, mobius, sphere_mobius
        print('lands source', lands_source)
        S_correspondences = torch.cat([lands_source[:3,:] +  0.0001  , lands_target[:3,:] + 0.0001],0) ##### Check if correct way around.
        
        mobius_sig = find_mobius_from_3corresp(S_correspondences)
        print('mobius information:', mobius_sig)

        print('testing mobius signature:')
        print('sphere mobius here we go', sphere_mobius(mobius_sig[0], lands_source[:3, :]) - torch.tensor([[0.0,0.0,-1.0],[1.0,0.0,0.0],[0.0,0.0,1.0]]))
        print('sphere mobius here we go', sphere_mobius(mobius_sig[1], torch.tensor([[0.0,0.0,-1.0],[1.0,0.0,0.0], [0.0,0.0,1.00001]])  ) - lands_target[:3, :])        
        #print('sphere mobius here we go', sphere_mobius(mobius_sig[1], lands_target[:3, :]) - torch.tensor([[0.0,0.0,-1.0],[1.0,0.0,0.0],[0.0,0.0,1.0]]))
        return mobius_sig



    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
        
        
        