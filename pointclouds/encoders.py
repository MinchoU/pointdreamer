import ninjax as nj
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
from .networks import PointConv

class PointConvEncoder(nj.Module):
  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  outer: bool = False
  strided: bool = False
  #### for rgbd
  depth_max: int = 1000
  #### for pointcloud
  # use_pcd: bool = False
#   bandwidths: tuple = (0.05,0.1)
#   npoints: tuple = (64, 32)
#   nsamples: tuple = (8, 16)
#   mlps: tuple = ((16, 32),(32, 64))
#   bandwidths: tuple = (0.05,0.1)
#   npoints: tuple = (512,256)
#   nsamples: tuple = (64, 128)
#   mlps: tuple = ((16, 32),(32, 64))
  bandwidths: tuple = (0.05, 0.1, 0.2, 0.4)
  npoints: tuple = (512, 256, 128, 64)
  nsamples: tuple = (16, 32, 64, 128)
  mlps: tuple = ((16, 32), (32, 64), (64, 128), (128, 256))
  checkpoint: tuple = () #(1,),
  pool: str = 'mean'

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    assert 'pointcloud' in self.obs_space, "No key named 'pointcloud' in obs_space. Use simple encoder instead."
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2 and k!='pointcloud']
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, obs, reset, training, single=False):
    bdims = 1 if single else 2
    outs = []
    bshape = reset.shape

    if self.veckeys:
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      vecs = {k: obs[k] for k in self.veckeys}
      squish = nn.symlog if self.symlog else lambda x: x
      x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
      x = x.reshape((-1, *x.shape[bdims:]))
      for i in range(self.layers):
        x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
      outs.append(x)

    if self.imgkeys:
      K = self.kernel
      imgs = [obs[k] for k in sorted(self.imgkeys)]
      assert all(x.dtype == jnp.uint8 or x.shape[-1]==4 for x in imgs)
      normed = []
      for img in imgs:
          if img.shape[-1] == 4:                      
              rgb, depth = img[...,:3], img[...,-1:]
              rgb   = rgb.astype(jnp.float32) / 255.0 
              depth = depth.astype(jnp.float32) / self.depth_max
              img   = jnp.concatenate([rgb, depth], axis=-1)
          else:
              img = img.astype(jnp.float32) / 255.0
          normed.append(img)

      x = nn.cast(jnp.concatenate(normed, -1), force=True) - 0.5   # → (-0.5, +0.5)

      x = x.reshape((-1, *x.shape[bdims:]))
      for i, depth in enumerate(self.depths):
        if self.outer and i == 0:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
        elif self.strided:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
        else:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
          B, H, W, C = x.shape
          x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
        x = nn.act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
      assert 3 <= x.shape[-3] <= 16, x.shape
      assert 3 <= x.shape[-2] <= 16, x.shape
      x = x.reshape((x.shape[0], -1))
      outs.append(x)

    points = obs['pointcloud'] # [B, T, N, C+D]s
    points = nn.cast(points)
    
    # [B, N, C] 또는 [B, T, N, C] 형태로 변환
    if len(points.shape) == bdims + 2:  # [B, N, C] or [B, T, N, C]
        xyz = points[..., :3]
        features = points[..., 3:] if points.shape[-1] > 3 else None
    else:
        raise ValueError(f"Unexpected point cloud shape: {points.shape}")
      
    xyz = xyz.reshape(-1, *xyz.shape[bdims:])
    if features is not None:
        features = features.reshape(-1, *features.shape[bdims:])

    for i in range(len(self.npoints)):
       xyz, features = self.sub(
            f'pointconv{i}',
            PointConv,
            npoint=self.npoints[i],
            nsample=self.nsamples[i],
            in_channel=xyz.shape[1] + (0 if features is None else features.shape[1]),
            mlp=self.mlps[i],
            bandwidth=self.bandwidths[i],
            group_all=False
        )(xyz, features, training)
      
    if self.pool == 'mean':
        pcd_features = jnp.mean(features, axis=-2)  # [B, mlp[-1]]
    else:
        raise NotImplementedError

    outs.append(pcd_features)

    x = jnp.concatenate(outs, -1)
    tokens = x.reshape((*bshape, *x.shape[1:]))
    entries = {}
    return carry, entries, tokens
