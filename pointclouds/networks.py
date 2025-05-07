import embodied.jax.nets as nn
import jax.numpy as jnp
import ninjax as nj
import jax
from .nn_utils import compute_density, sample_and_group, sample_and_group_all, _sample_and_group_jit
class DensityNet(nj.Module):
    """
    Density network for density weighting
    """
    
    def __init__(self, hidden_unit=[16, 8]):
        super().__init__()
        self.hidden_unit = hidden_unit
    
        # conv2d : B, H, W, Cin -> B, W, H, Cout
        # batchnorm2d : B, H, W, C

    def __call__(self, density_scale, train):
        """
        Input:
            density_scale: [B, N, nsample, 1]
        Output:
            density_scale: [B, N, nsample, 1]
        """
        x = self.sub('conv0', nn.Conv2D, self.hidden_unit[0], 1)(density_scale)
        x = self.sub('bn0', nn.BatchNorm2d, self.hidden_unit[0])(x, train)
        x = jax.nn.relu(x)
        
        for i in range(1, len(self.hidden_unit)):
            x = self.sub(f'conv{i}', nn.Conv2D, self.hidden_unit[i], 1)(x)
            x = self.sub(f'bn{i}', nn.BatchNorm2d, self.hidden_unit[i])(x, train)
            x = jax.nn.relu(x)
        
        x = self.sub('conv_out', nn.Conv2D, 1, 1)(x)
        x = self.sub('bn_out', nn.BatchNorm2d, 1)(x, train)
        x = jax.nn.sigmoid(x)

        return x

class WeightNet(nj.Module):
    """
    Weight network for learning point weights
    """
    
    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8]):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_unit = hidden_unit
    
    def __call__(self, localized_xyz, train):
        """
        Input:
            localized_xyz: [B, C, nsample, N]
        Output:
            weights: [B, out_channel, nsample, N]
        """
        weights = jnp.transpose(localized_xyz, (0,3,2,1))
        if self.hidden_unit is None or len(self.hidden_unit) == 0:
            weights = self.sub('conv0', nn.Conv2D, self.out_channel, 1)(weights)
            weights = self.sub('bn0', nn.BatchNorm2d, self.out_channel)(weights, train)
            weights = jax.nn.relu(weights)
        
        else:
            weights = self.sub('conv0', nn.Conv2D, self.hidden_unit[0], 1)(weights)
            weights = self.sub('bn0', nn.BatchNorm2d, self.hidden_unit[0])(weights, train)
            weights = jax.nn.relu(weights)

            for i in range(1, len(self.hidden_unit)):
                weights = self.sub(f'conv{i}', nn.Conv2D, self.hidden_unit[i], 1)(weights)
                weights = self.sub(f'bn{i}', nn.BatchNorm2d, self.hidden_unit[i])(weights, train)
                weights = jax.nn.relu(weights)
            
            weights = self.sub('conv_out', nn.Conv2D, self.out_channel, 1)(weights)
            weights = self.sub('bn_out', nn.BatchNorm2d, self.out_channel)(weights, train)
            weights = jax.nn.relu(weights)

        return weights
    
class PointConv(nj.Module):
    """
    PointConv with density estimation
    """
    
    def __init__(self, npoint, nsample, in_channel, mlp, bandwidth, group_all):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.bandwidth = bandwidth
        self.group_all = group_all
        self.in_channel = in_channel
        self.weight_net = WeightNet(3, 16, name='wnet')
        self.density_net = DensityNet(name='dnet')
    
    def __call__(self, xyz, points, train=True):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C_in]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points: sample points feature data, [B, S, C_out]
        """        
        # Compute density
        B, N, C = xyz.shape
        # xyz_density = nj.checkpoint(compute_density)(xyz, self.bandwidth)
        xyz_density = compute_density(xyz, self.bandwidth)
        inverse_density = 1.0 / (xyz_density+1e-6)
        
        # Sample and group
        def group(xyz, points, inv_den):
            if self.group_all:
                new_xyz, new_points, grouped_xyz_norm, grouped_density = sample_and_group_all(
                    xyz, points, jnp.expand_dims(inverse_density, -1)
                )
            else:
                new_xyz, new_points, grouped_xyz_norm, _, grouped_density = _sample_and_group_jit(
                    self.npoint, self.nsample, xyz, points, jnp.expand_dims(inverse_density, -1)
                )
        
            return new_xyz, new_points, grouped_xyz_norm, grouped_density

        new_xyz, new_points, grouped_xyz_norm, grouped_density = group(xyz, points, inverse_density)
        
        def mlp_block(x):
            for i, out_channel in enumerate(self.mlp):
                x = self.sub(f'mlp_conv{i}', nn.Conv2D, out_channel, 1)(x) # [B, npoint, nsample, outchannel]
                x = self.sub(f'mlp_bn{i}', nn.BatchNorm2d, out_channel)(x, train)
                x = jax.nn.relu(x)
            return x
        
        x = mlp_block(new_points)

        # Apply density weighting
        def density(x, grouped_density):
            inverse_max_density = jnp.max(grouped_density, axis=2, keepdims=True)
            density_scale = grouped_density / (inverse_max_density+1e-6)
            
            # Apply density network
            density_scale = self.density_net(density_scale, train)
        
            # Apply weights
            x = x * density_scale
            return x
        
        x = density(x, grouped_density)
        
        # Apply weight network
        def weight(x, grouped_xyz_norm):
            grouped_xyz = jnp.transpose(grouped_xyz_norm, (0, 3, 2, 1))
            weights = self.weight_net(grouped_xyz, train) # B*N*n_sample*C_mid
            # Matrix multiplication
            x = jnp.transpose(x, (0, 1, 3, 2))
            x = jnp.matmul(x,weights)
            return x
        
        x = weight(x, grouped_xyz_norm)
        
        # Reshape and apply linear layer (final 1*1 conv)
        x = x.reshape(B, self.npoint, -1)
        x = self.sub('conv_out', nn.Linear, self.mlp[-1])(x)
        x = self.sub('bn_out', nn.BatchNorm1d, self.mlp[-1])(x, train)
        x = jax.nn.relu(x)
        
        return new_xyz, x

