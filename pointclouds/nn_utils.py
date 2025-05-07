import jax
import jax.numpy as jnp
import ninjax as nj
import embodied.jax.nets as nn
from functools import partial
from typing import Optional, Tuple, List, Union
sg = jax.lax.stop_gradient

def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points.
    
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    
    src_squared = jnp.sum(src ** 2, -1)
    dst_squared = jnp.sum(dst ** 2, -1)
    dist = src_squared[:, :, None] - 2 * jnp.matmul(src, jnp.transpose(dst, (0, 2, 1))) + dst_squared[:, None, :]
    
    return sg(jnp.maximum(dist, 0.0))

# def index_points(points, idx):
#     # points: [B, N, C], idx: [B, S] or [B, S, K]
#     idx_exp = idx[..., None]  # [..., 1]
#     out = jnp.take_along_axis(points[:, None], idx_exp[..., None], axis=1)
#     return out.squeeze(-1)

def index_points(points, idx):
    """
    points: [B, N, C]
    idx: [B, S] 또는 [B, S, K]
    """
    if len(idx.shape) == 2:  # [B, S]
        idx_expanded = idx[..., None]  # [B, S, 1]
        return sg(jnp.take_along_axis(points, idx_expanded, axis=1))  # [B, S, C]
    else:  # [B, S, K]
        # 3D 인덱싱을 위한 다른 접근 방식
        B, N, C = points.shape
        _, S, K = idx.shape
        
        # 인덱스를 펼쳐서 2D 인덱싱으로 변환
        idx_flat = idx.reshape(B, -1)  # [B, S*K]
        
        # 2D 인덱싱 수행
        gathered_flat = index_points(points, idx_flat)  # [B, S*K, C]
        
        # 원래 형태로 복원
        return gathered_flat.reshape(B, S, K, C)  # [B, S, K, C]


# def index_points(points, idx):
#     """
#     points: [B, N, C]
#     idx:    [B, S] or [B, S, K]
#     returns:
#       if idx.ndim==2 -> [B, S, C]
#       if idx.ndim==3 -> [B, S, K, C]
#     """
#     def gather_fn(p, i):
#         # p: [N, C], i: [S] or [S, K]
#         return p[i]  # numpy‐style advanced indexing → [S, C] or [S, K, C]

#     # vmap over batch dimension
#     return jax.vmap(gather_fn, in_axes=(0, 0))(points, idx)

# def index_points(points, idx):
#     """
#     Input:
#         points: input points data, [B, N, C]
#         idx: sample index data, [B, S] or [B, S, K]
#     Return:
#         new_points: indexed points data, [B, S, C] or [B, S, K, C]
#     """
#     B = points.shape[0]
    
#     view_shape = list(idx.shape)
#     view_shape[1:] = [1] * (len(view_shape) - 1)
    
#     repeat_shape = list(idx.shape)
#     repeat_shape[0] = 1
    
#     batch_indices = jnp.tile(jnp.reshape(jnp.arange(B, dtype=jnp.int32), view_shape), repeat_shape)
    
#     # Handle different idx shapes
#     if len(idx.shape) == 2:
#         return points[batch_indices, idx]
#     elif len(idx.shape) == 3:
#         return points[batch_indices, idx]
#     else:
#         raise ValueError("Invalid idx shape")


def farthest_point_sample(points, num_points, check_val=False):
    """
    input:
        points: pointcloud data, [B, N, C]
        num_points: number of samples
        check_val: if True, xyzw input
    Return

    To avoid cumbersome random seed, select the first point as the most center point.
    """
    def fps_one(p):
        if check_val:
            coords_before = p[:, :3]
            mask = p[:, 3] > 0
            coords_after = p[:, 4:]
            
            if coords_after.shape[1] > 0:
                coords = jnp.concatenate([coords_before, coords_after], axis=1)
            else:
                coords = coords_before
        else:
            coords = p
            coords_before = p[:, :3]
            coords_after = p[:, 3:]
            mask = jnp.full(coords.shape[0], True, dtype=bool)

        centroid = jnp.mean(coords, axis=0)
        center_dists = jnp.sum((coords - centroid)**2, axis=1)
        first_idx = jnp.argmin(center_dists)

        N = coords.shape[0]
        centroids = jnp.zeros(num_points, jnp.int32)

        distances = jnp.full(N, -jnp.inf)
        distances = jnp.where(mask, jnp.full(N, jnp.inf), distances)
        centroids = centroids.at[0].set(first_idx)
        
        c = coords_before[first_idx]
        diff = coords_before - c
        dist2 = jnp.sum(diff * diff, axis=1)
        
        dist2 = jnp.where(mask, dist2, jnp.inf)
        distances = jnp.minimum(distances, dist2)
        
        selected_mask = jnp.zeros(N, dtype=bool).at[first_idx].set(True)
        distances = jnp.where(selected_mask, -jnp.inf, distances)
        
        def body_fn(i, state):
            centroids, distances = state
            next_idx = jnp.argmax(distances)
            centroids = centroids.at[i].set(next_idx)
            c = coords_before[next_idx]

            dist2 = jnp.sum((coords_before - c)**2, axis=1)
            dist2 = jnp.where(mask, dist2, jnp.inf)
            distances = jnp.minimum(distances, dist2)
            distances = distances.at[next_idx].set(-jnp.inf)
            return centroids, distances

        centroids, _ = jax.lax.fori_loop(1, num_points, body_fn,
                                         (centroids, distances))
        sampled_coords = coords[centroids]   # [num_points, C]
        return sg(centroids), sg(sampled_coords)

    indices, coords = jax.vmap(fps_one)(points)
    return indices, coords

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    group_idx = jnp.tile(jnp.reshape(jnp.arange(N, dtype=jnp.int32), (1, 1, N)), (B, S, 1))
    sqrdists = square_distance(new_xyz, xyz)
    
    # Mark points outside radius with N
    group_idx = jnp.where(sqrdists > radius ** 2, N, group_idx)
    
    # Sort and take first nsample points
    group_idx = jnp.sort(group_idx, axis=-1)
    group_idx = jax.lax.dynamic_slice_in_dim(group_idx, 0, nsample, axis=-1)
    
    # Handle case where not enough points in radius
    group_first = jnp.tile(group_idx[:, :, 0:1], (1, 1, nsample))
    mask = group_idx == N
    group_idx = jnp.where(mask, group_first, group_idx)

    return sg(group_idx)


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = jax.lax.top_k(-sqrdists, nsample)
    return sg(group_idx)


def sample_and_group(npoint, nsample, xyz, points, density_scale=None):
    """
    Input:
        npoint: number of points to sample
        nsample: number of neighbors to group
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
        density_scale: density scale, [B, N, 1]
    Return:
        new_xyz: sampled points position data, [B, npoint, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
        grouped_xyz_norm: grouped xyz normalized, [B, npoint, nsample, C]
        idx: indices of grouped points, [B, npoint, nsample]
        grouped_density: grouped density, [B, npoint, nsample, 1]
    """
    B, N, C = xyz.shape
    S = npoint
    
    # Sample points
    fps_idx, _ = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C]
    
    # Group points
    idx = knn_point(nsample, xyz, new_xyz)  # [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - jnp.expand_dims(new_xyz, 2)  # [B, npoint, nsample, C]
    
    if points is not None:
        grouped_points = index_points(points, idx)  # [B, npoint, nsample, D]
        new_points = jnp.concatenate([grouped_xyz_norm, grouped_points], axis=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    
    if density_scale is None:
        return new_xyz, new_points, grouped_xyz_norm, idx
    else:
        grouped_density = index_points(density_scale, idx)  # [B, npoint, nsample, 1]
        return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density


def sample_and_group_all(xyz, points, density_scale=None):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
        density_scale: density scale, [B, N, 1]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
        grouped_xyz: grouped xyz, [B, 1, N, C]
        grouped_density: grouped density, [B, 1, N, 1]
    """
    B, N, C = xyz.shape
    
    # Use mean as center
    new_xyz = jnp.mean(xyz, axis=1, keepdims=True)  # [B, 1, C]
    
    # Group all points
    grouped_xyz = jnp.expand_dims(xyz, 1) - jnp.expand_dims(new_xyz, 2)  # [B, 1, N, C]
    
    if points is not None:
        new_points = jnp.concatenate([grouped_xyz, jnp.expand_dims(points, 1)], axis=-1)  # [B, 1, N, C+D]
    else:
        new_points = grouped_xyz
    
    if density_scale is None:
        return new_xyz, new_points, grouped_xyz
    else:
        grouped_density = jnp.expand_dims(density_scale, 1)  # [B, 1, N, 1]
        return new_xyz, new_points, grouped_xyz, grouped_density


def compute_density(xyz, bandwidth):
    """
    Compute density for points
    
    Input:
        xyz: input points position data, [B, N, C]
        bandwidth: bandwidth for density estimation
    Return:
        density: density for each point, [B, N]
    """
    B, N, C = xyz.shape
    sqrdists = square_distance(xyz, xyz)
    gaussian_density = jnp.exp(-sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = jnp.mean(gaussian_density, axis=-1)
    
    return xyz_density

_fps_jit = partial(jax.jit, static_argnums=(1,2))(farthest_point_sample)

@partial(jax.jit, static_argnums=(0,1))
def _sample_and_group_jit(npoint, nsample, xyz, points, density_scale=None):
    # B, N, C = xyz.shape
    # S = npoint
    fps_idx, _ = _fps_jit(xyz, npoint)             # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)           # [B, npoint, C]

    idx = knn_point(nsample, xyz, new_xyz)         # [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)           # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None]

    if points is not None:
        grouped_points = index_points(points, idx) # [B, npoint, nsample, D]
        new_points = jnp.concatenate([grouped_xyz_norm, grouped_points], -1)
    else:
        new_points = grouped_xyz_norm

    if density_scale is None:
        return new_xyz, new_points, grouped_xyz_norm, idx, None
    else:
        gd = index_points(density_scale, idx)       # [B, npoint, nsample, 1]
        return new_xyz, new_points, grouped_xyz_norm, idx, gd