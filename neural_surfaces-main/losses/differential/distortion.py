
import torch

#torch.autograd.set_detect_anomaly(True)

from differential import DifferentialModule
from torch.nn import functional as F

class DistortionMixin(DifferentialModule):

    ## Is there a way to avoid this? i.e. not calling this
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.register_buffer('zero', torch.tensor(0.0))
        self.register_buffer('eye', torch.eye(2))
        self.register_buffer('eps',  torch.tensor(1.0e-6))
        self.register_buffer('one',  torch.tensor(1.0))


    def conformal1(self, FFF):
        E = FFF[:, 0,0]
        G = FFF[:, 1,1]

        ### conformal: || _lambda * M - I ||
        lambd = (E + G) / FFF.pow(2).sum(-1).sum(-1)
        
        #print('lamd',lambd.shape)
        #print('FFF',FFF.shape)
        
        ppd   = (lambd.view(-1, 1, 1) * FFF - self.eye).pow(2).sum(-1).sum(-1)
        return ppd

    def conformal_AMIPS_2D(self, FFF):
        E = FFF[:, 0,0]
        G = FFF[:, 1,1]
        F = FFF[:, 0,1]

        ### conformal: || _lambda * M - I ||
        ppd =   (E + G)/torch.sqrt(E*G - F**2 + self.eps) 
         
        return ppd
                       
    def symmetric_dirichlet(self, FFF):
        ## print('shape',(FFF + self.eps*self.eye).shape)
        #print(FFF.dtype)
        FFtarget_inv = (FFF + self.eps*self.eye).inverse()

        E = FFF[:, 0,0] - 1.0
        G = FFF[:, 1,1] - 1.0
        E_inv = FFtarget_inv[:, 0,0] - 1.0
        G_inv = FFtarget_inv[:, 1,1] - 1.0
        #Is this correct?

        ### symm dirichlet: trace(J^T J) + trace(J_inv^T J_inv )
        dirichlet = E + G
        inv_dirichlet = E_inv + G_inv
        ppd = dirichlet + inv_dirichlet
        return ppd


    def arap(self, FFF):
        ppd = (FFF - self.eye).pow(2).sum(-1).sum(-1)
        return ppd


    def equiareal(self, FFF): #Romy: fixed from EG - 1 to EG - F^2 - 1
        E = FFF[:, 0,0]
        F = FFF[:,0,1]
        G = FFF[:, 1,1]

        ### equi-areal: E*G _ F^2 = 1
        ppd = (E * G) - F.pow(2) - 1
        ppd = ppd.pow(2) + (1.0 / (ppd + self.eps)).pow(2)

        return ppd


    def fold_regularization(self, J):
        J_det = J.det()
        if torch.min(J_det)<0:
            print('fold detected')
        J_det_sign = torch.sign(J_det)
        pp_fold = 1.0*    torch.max(-J_det_sign * torch.exp(-J_det), self.zero)      #######I'm turning it on again!!!

        return pp_fold
        

    
    
    
    def compute_differential_quantities(self, target_points3D, target_points_sphere, source_points_sphere, source_points3D):
        
        
        ### Note: The main purpose of this function to to compute a 2x2 matrix that measures the 'stretch' between corresponding patches
        ### on the source and target surface. I have called this matrix the FFF (first fundamental form) but that is a bit of an abuse of notation.
         
        '''
        target_points3D : Nx3 3D points on target surface
        target_points2D : Nx2 2D points in target surface domain           ########outdated
        source_points2D : Nx2 2D points in source surface domain
        source_points3D : Nx3 3D points on source surface
        '''
        
        #print('shapes', target_points_sphere.shape, source_points_sphere.shape, target_points3D.shape, source_points3D.shape )
        for i in range(1):
           
		
            J_psi_3x3  = self.gradient(out=target_points3D, wrt=target_points_sphere) #3x3, T surface wrt T sphere
            #print('shapes', J_psi_3x3.shape )
            
            J_h_3x3  = self.gradient(out=target_points_sphere, wrt=source_points_sphere)         

            #print('shapes', J_psi_3x3.shape )            
            ### out3,wrt3=source_points3D, source_points2D#by 2D I mean sphere S^2 . Legacy.
            J_phi_3x3     = self.gradient(out=source_points3D, wrt=source_points_sphere)
   
            
            ##### TO DO: copy over idea from differential module. 
            
            u=torch.acos(source_points_sphere[:,2]) ###### arccos(z) between 0 and pi
            v=torch.atan2(source_points_sphere[:,1], source_points_sphere[:,0]) ####
            
            ## this is J (r) i.e. Jacobian of a local sphere parametrisation. 
            dg_du = torch.transpose (torch.stack((torch.cos(u)*torch.cos(v),torch.cos(u)*torch.sin(v),  -1.0*torch.sin(u) ) ), 0, 1 )             
            dg_dv = torch.transpose ( torch.stack((-1.0*torch.sin(v), torch.cos(v),  0.0*u  )) , 0, 1 )
            
            dg_du = dg_du.unsqueeze(-1)
            dg_dv = dg_dv.unsqueeze(-1)
            
            ### below block for folding loss only          
            #dh_du = torch.einsum('ijk,ik->ij',J_h_3x3, dg_du)
            #dh_dv = torch.einsum('ijk,ik->ij',J_h_3x3, dg_dv)
            print(J_h_3x3.shape, dg_du.shape)
            dh_du = J_h_3x3 @ dg_du
            dh_dv = J_h_3x3 @ dg_dv
            
            
            
            
            vol_mat = torch.stack((dh_du.squeeze(), dh_dv.squeeze(), target_points_sphere)).permute(1,2,0) # used for folding loss
            ### above block for folding loss only
		
		
		
		    ### find two vectors that lie in the tangent plane of source surface
            #dx_du = torch.einsum('ijk,ik->ij',J_phi_3x3, dg_du)
            #dx_dv = torch.einsum('ijk,ik->ij',J_phi_3x3, dg_dv)
            dx_du = J_phi_3x3 @ dg_du
            dx_dv = J_phi_3x3 @ dg_dv
            
            ### find normal to source surface
            n = torch.cross(dx_du, dx_dv)
            n = F.normalize(n,p=2,dim=1)
            
            ### construct orthonormal basis for tangent plane
            ###normalise dx_du for one basis vector.
            b1 = F.normalize(dx_du, p=2,dim=1)
            b2 = F.normalize(torch.cross(n, b1), p=2, dim=1).squeeze()
            #b2 = (b2.T/(torch.einsum('ij,ij->i',b2, b2)**0.5).T).T ###normalise n cross b1 for other basis vector.
            b1=b1.squeeze()            

            #print('basis shape', b1.shape, b2.shape)            
             
            R = torch.stack((b1, b2)).permute(1,2,0)
            #print('dim R is',R.shape)
            #print('dim psi, dim Jh, dim phi, is ',J_psi_3x3.shape, J_h_3x3.shape, J_phi_3x3.shape)
            
            J = J_psi_3x3 @ J_h_3x3 @ J_phi_3x3.inverse() @ R
            
            
                     
            # First Fundamental Form
            ### This 2x2 matrix  measures the 'stretch' between corresponding patches on the source and target surface.
            ### I have called this matrix the FFF (first fundamental form) but that is a bit of an abuse of notation.
            
            FFF = J.transpose(1,2).matmul(J)### (2x3) x (3x2) = (2x2)
            
            
            
        return FFF, vol_mat  ### vol_mat (previously Jh) is used in the folding loss only.
        
