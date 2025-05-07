import ninjax as nj
import jax
import jax.numpy as jnp
from .nn_utils import farthest_point_sample

class FPS(nj.Module):
    def __init__(self, num_points, **kw):
        self.num_points = num_points
        for key, val in kw.items():
            setattr(self, key, val)

    def __call__(self, points):
        return farthest_point_sample(points, self.num_points, True)[1]