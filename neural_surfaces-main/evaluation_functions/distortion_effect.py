import numpy as np
import trimesh
from matplotlib import pyplot as plt


######SNS stuff######
from differential import *
from differential.batches_diff_quant import batches_diff_quant as bdq

from runners import MainRunner
from mains.experiment_configurator import ExperimentConfigurator
import torch
#####################


def replace_nans(arr, value=0.0):
    """Replace NaNs in an array with a given value."""
    arr[np.isnan(arr)] = value
    return arr


def compute_jacobians(mesh_3d, mesh_param, faces):
    """
    Computes the Jacobian matrices for a parameterization between two meshes.
    
    Args:
        mesh_3d (numpy.ndarray): n x 3 array of 3D vertices (original mesh).
        mesh_param (numpy.ndarray): n x 3 array of 3D vertices (parameterized mesh).
        faces (numpy.ndarray): m x 3 array of triangle vertex indices.
    
    Returns:
        numpy.ndarray: An m x 2 x 2 array of Jacobian matrices.
    """
    print("Mesh 3D shape:", mesh_3d.shape)
    print("Mesh param shape:", mesh_param.shape)
    print("Faces shape:", faces.shape)

    jacobians = []

    for face in faces:
        # Get indices of the vertices forming the triangle
        i1, i2, i3 = face

        # Get 3D vertices of the triangle
        v1, v2, v3 = mesh_3d[i1], mesh_3d[i2], mesh_3d[i3]

        # Get parameterized vertices of the triangle
        u1, u2, u3 = mesh_param[i1], mesh_param[i2], mesh_param[i3]

        # Compute edge vectors in 3D
        e1 = v2 - v1
        e2 = v3 - v1

        # Compute edge vectors in parameterized space
        f1 = u2 - u1
        f2 = u3 - u1

        # Construct edge matrices
        #F = np.array([f1[:2], f2[:2]]).T  # 2x2 (parameterized space)
        E = np.array([e1, e2])
        F = np.array([f1, f2])

        # Compute Jacobian using least-squares approximation
        J = F @ np.linalg.pinv(E)
        jacobians.append(J)

    return np.array(jacobians)

def compute_conformal_distortion(jacobians):
    """
    Computes the conformal distortions for all triangles in a parameterized mesh.

    Args:
        jacobians (numpy.ndarray): m x 2 x 2 array of Jacobian matrices.

    Returns:
        numpy.ndarray: An array of conformal distortions for each triangle.
    """
    # Singular Value Decomposition to get singular values
    singular_values = np.linalg.svd(jacobians, compute_uv=False)

    # Extract maximum and minimum singular values
    sigma_max = singular_values[:, 0]  # Largest singular value
    sigma_min = singular_values[:, 1]  # Smallest singular value

    # Compute conformal distortion
    conformal_distortions = sigma_max / sigma_min
    return conformal_distortions



def compute_area_distortion_vectorized(jacobians):
    """
    Computes the area distortions for all triangles in a vectorized manner.

    Args:
        jacobians (numpy.ndarray): m x 2 x 2 array of Jacobian matrices.

    Returns:
        numpy.ndarray: An array of area distortions for each triangle.
    """
    # Compute the determinant of each 2x2 Jacobian matrix
    det_j = np.linalg.det(jacobians)

    # Area distortion: Absolute value of determinant minus 1
    area_distortions = np.abs(det_j) - 1
    return area_distortions


# Load the original mesh
#mesh = trimesh.load('../data/treefrog11657.obj')
mesh = trimesh.load('../data/FLOWER.obj')

# Loop through parameterized variants
for variant in ['original', 'var1', 'var2', 'var4', 'var5', 'var6']:
    param = trimesh.load(f'../data/distortion_expmt/{variant}.obj')
    print(f"Processing variant: {variant}")

    # Compute Jacobians
    J = compute_jacobians(mesh.vertices, param.vertices, mesh.faces)

    # Compute area distortions
    area_distortions = compute_area_distortion_vectorized(J)
    angle_distortions = compute_conformal_distortion(J)

    # Output results
    print(f"Variant: {variant}")
    #print(f"Jacobian matrices shape: {J.shape}")
    print('avg area distortion', area_distortions.mean())
    print('avg angle distortion', angle_distortions.mean())
    print("-" * 50)

    '''
    plt.plot(area_distortions)
    plt.title('area distortions')
    plt.show()

    plt.plot(angle_distortions)
    plt.title('angle distortions')
    plt.show()
    '''
