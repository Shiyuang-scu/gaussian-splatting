import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn

class GaussianModel:

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.significance_score = torch.empty(0)
        
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.covariance_activation = self.build_covariance_from_scaling_rotation

    def save_ply(self, path):

        xyz = self._xyz.detach().cpu().numpy()
        # normals = np.zeros_like(xyz)
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        normals = np.stack((np.asarray(plydata.elements[0]["nx"]),
                            np.asarray(plydata.elements[0]["ny"]),
                            np.asarray(plydata.elements[0]["nz"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scales = self._extracted_from_load_ply_26(plydata, "scale_", xyz)
        rots = self._extracted_from_load_ply_26(plydata, "rot", xyz)
        self._xyz = torch.tensor(xyz, dtype=torch.float, device="cpu")
        self._normal = torch.tensor(normals, dtype=torch.float, device="cpu")
        self._features_dc = torch.tensor(features_dc, dtype=torch.float, device="cpu").transpose(1, 2).contiguous()
        self._features_rest = torch.tensor(features_extra, dtype=torch.float, device="cpu").transpose(1, 2).contiguous()
        self._opacity = torch.tensor(opacities, dtype=torch.float, device="cpu")
        self._scaling = torch.tensor(scales, dtype=torch.float, device="cpu")
        self._rotation = torch.tensor(rots, dtype=torch.float, device="cpu")

        self.active_sh_degree = self.max_sh_degree

    def _extracted_from_load_ply_26(self, plydata, arg1, xyz):
        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith(arg1)
        ]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        result = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            result[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return result

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        l.extend(
            f'f_dc_{i}'
            for i in range(self._features_dc.shape[1] * self._features_dc.shape[2])
        )
        l.extend(
            f'f_rest_{i}'
            for i in range(
                self._features_rest.shape[1] * self._features_rest.shape[2]
            )
        )
        l.append('opacity')
        l.extend(f'scale_{i}' for i in range(self._scaling.shape[1]))
        l.extend(f'rot_{i}' for i in range(self._rotation.shape[1]))
        return l
    

    def build_rotation(self, r):
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

        q = r / norm[:, None]

        R = torch.zeros((q.size(0), 3, 3), device='cpu')

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        return R
    
    def build_scaling_rotation(self, s, r):
        L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cpu")
        R = self.build_rotation(r)

        L[:,0,0] = s[:,0]
        L[:,1,1] = s[:,1]
        L[:,2,2] = s[:,2]

        L = R @ L
        return L
    
    def strip_lowerdiag(self, L):
        uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cpu")

        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 0, 2]
        uncertainty[:, 3] = L[:, 1, 1]
        uncertainty[:, 4] = L[:, 1, 2]
        uncertainty[:, 5] = L[:, 2, 2]
        return uncertainty

    def strip_symmetric(self, sym):
        return self.strip_lowerdiag(sym)
    
    def build_covariance_from_scaling_rotation(self, scaling, scaling_modifier, rotation):
        L = self.build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        return self.strip_symmetric(actual_covariance)
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)


def read_line3d_from_txt(line3d_path):
    with open(line3d_path, 'r') as f:
        lines = f.readlines()
        # for each line, read the 2nd to 7th values as the line direction, store them in the format of ((a,b,c),(d,e,f))
        line_vertices = np.array([
            [
                [float(line.split()[1]), float(line.split()[2]), float(line.split()[3])], 
                [float(line.split()[4]), float(line.split()[5]), float(line.split()[6])]
            ] 
            for line in lines])
    return line_vertices

def point_to_line_distance(point, line_start, line_end):
    """
    Calculate the minimum distance between a point and a line segment in 3D.

    Args:
    - point: A 3D point (numpy array).
    - line_start: Start point of the line segment (numpy array).
    - line_end: End point of the line segment (numpy array).

    Returns:
    - The distance between the point and the line segment.
    """
    # return np.linalg.norm( np.cross(line_end-line_start, line_start-point)) / np.linalg.norm(line_end-line_start)
    
    # https://stackoverflow.com/questions/54442057/calculate-the-euclidian-distance-between-an-array-of-points-to-a-line-segment-in/54442561#54442561
    d = np.divide(line_end - line_start, np.linalg.norm(line_end - line_start)) # normalized tangent vector
    
    s = np.dot(line_start - point, d) # signed parallel distance components
    t = np.dot(point - line_end, d) # signed parallel distance components

    h = np.maximum.reduce([s, t, 0]) # clamped parallel distance

    c = np.cross(point - line_start, d) # perpendicular distance component

    return np.hypot(h, np.linalg.norm(c))


def merge_splats(mu_array, cov_array, weights=None):
    if weights is None:
        weights = np.ones(len(mu_array))
    mu_merge = np.average(mu_array, axis=0, weights=weights)

    cov_merge_array = [
        cov_array[i]
        + np.matmul((mu_array[i] - mu_merge), (mu_array[i] - mu_merge).T)
        for i in range(len(mu_array))
    ]
    cov_merge = np.average(np.array(cov_merge_array), axis=0, weights=weights)

    return mu_merge, cov_merge


# restore a symmetric covariance matrix, based on the upper triangular matrix
def restore_covariance(cov):
    cov_matrix = np.zeros((3,3))
    cov_matrix[np.triu_indices(3)] = cov
    cov_matrix[np.tril_indices(3, -1)] = cov_matrix.T[np.tril_indices(3, -1)]
    return cov_matrix

def get_cov_ellipsoid(cov, mu=np.zeros((3)), nstd=3):
    """
    Return the 3d points representing the covariance matrix
    cov centred at mu and scaled by the factor nstd.

    Plot on your favourite 3d axis. 
    Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.1)
    Example 2:  ax.plot_surface(X,Y,Z,alpha=0.1)
    """
    assert cov.shape==(3,3)

    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov,axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    # Set of all spherical angles to draw our ellipsoid
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = nstd * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = X.shape
    # Flatten to vectorise rotation
    X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
    X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
    X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
   
    # Add in offsets for the mean
    X = X + mu[0]
    Y = Y + mu[1]
    Z = Z + mu[2]
    
    return X,Y,Z

def merge_two_3dgs(gs1_path, gs2_path):
    gs1 = GaussianModel(sh_degree=3)
    gs1.load_ply(gs1_path)
    position1 = gs1._xyz.detach().cpu().numpy()
    features_dc1 = gs1._features_dc.detach().cpu().numpy()
    features_rest1 = gs1._features_rest.detach().cpu().numpy()
    opacity1 = gs1._opacity.detach().cpu().numpy()
    scaling1 = gs1._scaling.detach().cpu().numpy()
    rotation1 = gs1._rotation.detach().cpu().numpy()
    
    gs2 = GaussianModel(sh_degree=3)
    gs2.load_ply(gs2_path)
    position2 = gs2._xyz.detach().cpu().numpy()
    features_dc2 = gs2._features_dc.detach().cpu().numpy()
    features_rest2 = gs2._features_rest.detach().cpu().numpy()
    opacity2 = gs2._opacity.detach().cpu().numpy()
    scaling2 = gs2._scaling.detach().cpu().numpy()
    rotation2 = gs2._rotation.detach().cpu().numpy()
    
    gs_merged = GaussianModel(sh_degree=3)
    gs_merged._xyz = nn.Parameter(torch.tensor(np.concatenate([position1, position2]), dtype=torch.float, device="cpu"))
    gs_merged._features_dc = nn.Parameter(torch.tensor(np.concatenate([features_dc1, features_dc2]), dtype=torch.float, device="cpu"))
    gs_merged._features_rest = nn.Parameter(torch.tensor(np.concatenate([features_rest1, features_rest2]), dtype=torch.float, device="cpu"))
    gs_merged._opacity = nn.Parameter(torch.tensor(np.concatenate([opacity1, opacity2]), dtype=torch.float, device="cpu"))
    gs_merged._scaling = nn.Parameter(torch.tensor(np.concatenate([scaling1, scaling2]), dtype=torch.float, device="cpu"))
    gs_merged._rotation = nn.Parameter(torch.tensor(np.concatenate([rotation1, rotation2]), dtype=torch.float, device="cpu"))

    return gs_merged

