
import numpy as np
import torch
import trimesh

## remove unreferenced vertices and compute normals
def clean_mesh(V, F):
    import pymeshlab

    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(V, F)
    mesh.add_vertex_custom_scalar_attribute(np.arange(V.shape[0]), 'idx')
    ms.add_mesh(mesh)
    ms.remove_unreferenced_vertices()
    ms.re_orient_all_faces_coherentely()
    ms.re_compute_vertex_normals()

    mesh = ms.current_mesh()

    V_small  = mesh.vertex_matrix()
    F_small  = mesh.face_matrix()
    N_small  = mesh.vertex_normal_matrix()
    NF_small = mesh.face_normal_matrix()
    V_idx = mesh.vertex_custom_scalar_attribute_array('idx').astype(np.int64)

    return V_small, F_small, N_small, V_idx, NF_small

def readOBJ(filename):

    V = []
    F = []
    UV = []
    TF = []
    N = []

    with open(filename, 'r') as stream:
        for line in stream.readlines():
            els = line.split(' ')
            els = list(filter(None, els))
            if els[0] == 'v':
                V.append([float(els[1]), float(els[2]), float(els[3])])
            elif els[0] == 'vt':
                UV.append([float(els[1]), float(els[2])])
            elif els[0] == 'vn':
                N.append([float(els[1]), float(els[2]), float(els[3])])
            elif els[0] == 'f':
                face = []
                face_uv = []
                #face

                for Fi in els[1:]:
                    F_els = Fi.split('/')
                    face.append(int(F_els[0]))
                    if len(F_els) > 1:
                        if len(F_els[1]) > 0:
                            face_uv.append(int(F_els[1]))
                        else:
                            face_uv.append(int(F_els[0]))

                F.append(face)
                TF.append(face_uv)

    V = np.array(V)
    F = np.array(F) - 1
    UV = np.array(UV)
    TF = np.array(TF) - 1
    N = np.array(N)

    return V, F, UV, TF, N

def writeOBJ(filename, V, F, UV, N):

    with open(filename, 'w') as stream:
        for i in range(V.shape[0]):
            stream.write('v {} {} {}\n'.format(V[i,0], V[i,1], V[i,2]))
            if UV is not None:
                stream.write('vt {} {}\n'.format(UV[i,0], UV[i,1]))
            if N is not None:
                stream.write('vn {} {} {}\n'.format(N[i,0], N[i,1], N[i,2]))

        for f in F:
            stream.write('f {}/{}/{} {}/{}/{} {}/{}/{}\n'.format(f[0]+1, f[0]+1, f[0]+1,
                                        f[1]+1, f[1]+1, f[1]+1, f[2]+1, f[2]+1, f[2]+1))
