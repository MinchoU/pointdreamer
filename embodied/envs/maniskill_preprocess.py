import functools

import elements
import embodied
import gymnasium as gym
import mani_skill
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
import numpy as np
import torch
from torch_geometric.nn import fps
import open3d as o3d

import numpy as np
import numpy as np

class FromManiskill(embodied.Env):
  def __init__(self, env, obs_mode='pointcloud+rgb', 
               control_mode=None, num_envs=1, size=(128,128), 
               cam_name='base_camera',
               depth_max_mm=1000,
               num_points=1024,
               point_process_strategy='fps',
               sim_backend='physx_cpu'
               ):
    # TODO seeding for reproducibility
    '''
    obs_mode : See https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html
      'pointcloud', 'rgb+depth', 'state_dict', 'state' (flattened), 'sensor_data' 
    control_mode : None will indicate pd_joint_delta_pos in PegInsertionSide-v1, 
      ex) 'pd_joint_delta_pos', 'pd_joint_pos', 'pd_ee_delta_pos', 'pd_ee_delta_pose', 
      'pd_ee_pose', 'pd_joint_target_delta_pos', 'pd_ee_target_delta_pos', 'pd_ee_target_delta_pose', 
      'pd_joint_vel', 'pd_joint_pos_vel', 'pd_joint_delta_pos_vel'
    num_envs : number of environments to create (maniskill parallelization). Currently only 1 env, relying on dreamer parallelization.
  
    '''
    self.use_colored_pcd = False
    if obs_mode == 'pointcloud+rgb':
      self.use_colored_pcd = True
      obs_mode = 'pointcloud'
    self._env = gym.make(env,
                         obs_mode=obs_mode,
                         control_mode=control_mode,
                         num_envs=num_envs,
                         sensor_configs=dict(width=size[0], height=size[1]),
                         sim_backend=sim_backend
                         )
    self.iscuda = sim_backend == 'physx_cuda'
    if not self.iscuda:
      self._env = CPUGymWrapper(self._env)



    self._obs_mode = obs_mode
    # self._env = CPUGymWrapper(self._env) # Make this parallel?

    self.cam_name = cam_name
    self.depth_max_mm = depth_max_mm

    self.num_points = num_points
    self.point_process_strategy = point_process_strategy

    if type(self.cam_name) != str and "rgb" in obs_mode:
      raise ValueError("Only single camera is supported for RGB observation.")
      # for pointclouds, data is automatically gained data using all cameras
      # , so need to modify default setup
    self._obs_key = obs_mode

    self._act_key = 'action'

    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._done = True
    self._info = None

  def to_np(self, tensor):
    if self.iscuda:
        return tensor.cpu().numpy()
    else:
      return tensor

  @property
  def env(self):
    return self._env

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    spaces = {}
    if 'state' in self._obs_key:
      # for key, value in self._env.observation_space['agent'].spaces.items():
      tot_dim = sum([self._env.observation_space['agent'][k].shape[0] for k in self._env.observation_space['agent'].keys()])
      spaces['state'] = elements.Space(np.float32, (tot_dim,))
  
    if 'pointcloud' in self._obs_key:
      pcd_dim = 6 if self.use_colored_pcd else 3
      spaces["pointcloud"] = elements.Space(np.float32, (self.num_points, pcd_dim))
      # spaces['pointcloud'] = elements.Space(np.float32, (self._env.observation_space['pointcloud']['xyzw'].shape[0], pcd_dim))

    if 'rgb' in self._obs_key:
      if 'depth' not in self._obs_key:
        spaces['image'] = elements.Space(np.uint8, self._env.observation_space['sensor_data'][self.cam_name]['rgb'].shape)
      else:
        spaces['image'] = elements.Space(np.uint16, tuple(list(self._env.observation_space['sensor_data'][self.cam_name]['rgb'].shape[:-1]) + [4]))

    return {
        **spaces,
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = elements.Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs, self._info = self._env.reset() 
      # info : 'elapsed_steps', 'success', 'peg_head_pos_at_hole', 'reconfigure'
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]
    obs, reward, terminated, truncated, self._info = self._env.step(action)
    reward = self.to_np(reward)
    terminated = self.to_np(terminated)
    truncated = self.to_np(truncated)

    self._done = terminated or truncated

    return self._obs(
        obs, reward,
        is_last=bool(truncated),
        is_terminal=bool(terminated))

  def process_pcd(self, points):
    """
    Process point cloud data to return exactly self.num_points points and corresponding indices.
    
    Args:
        points: Input point cloud data with shape (N, D), where D >= 4
              Can be numpy array or torch.Tensor depending on self.iscuda
        
    Returns:
        tuple: (processed_points, indices) 
            - processed_points: exactly self.num_points processed points (same type as input)
            - indices: indices of original points corresponding to processed points
    """
    if self.iscuda:
      # Remove invalid points (those with 4th dimension <= 0)
      valid_mask = points[:, 3] > 0
      valid_points = points[valid_mask]
      valid_indices = torch.where(valid_mask)[0]
      
      num_valid = valid_points.shape[0]
      
      if num_valid == 0:
        # Handle edge case with no valid points
        # Create dummy points and indices
        processed_points = torch.zeros((self.num_points, points.shape[1]), device=points.device, dtype=points.dtype)
        indices = torch.zeros(self.num_points, device=points.device, dtype=torch.long)
        return processed_points, indices
          
      if num_valid < self.num_points:
        # Need to augment by sampling
        # Calculate how many additional points we need
        num_needed = self.num_points - num_valid
        
        # Sample with replacement from valid points
        sample_indices = torch.randint(0, num_valid, (num_needed,), device=points.device)
        
        # Get the sampled points and their original indices
        augmented_points = valid_points[sample_indices]
        augmented_indices = valid_indices[sample_indices]
        
        # Combine original valid points with augmented points
        processed_points = torch.cat([valid_points, augmented_points], dim=0)
        indices = torch.cat([valid_indices, augmented_indices])
          
      else:
        # We have more valid points than needed, need to downsample
        if self.point_process_strategy == 'fps':
            # Using GPU-based farthest point sampling
            # Assuming fps returns indices
            fps_indices = fps(valid_points, self.num_points)
            processed_points = valid_points[fps_indices]
            indices = valid_indices[fps_indices]
        else:
            # Random sampling if not using FPS
            perm = torch.randperm(num_valid, device=points.device)[:self.num_points]
            processed_points = valid_points[perm]
            indices = valid_indices[perm]
        
    else:  # NumPy implementation
        
      # Remove invalid points (those with 4th dimension <= 0)
      valid_mask = points[:, 3] > 0
      valid_points = points[valid_mask]
      valid_indices = np.where(valid_mask)[0]
      
      num_valid = valid_points.shape[0]
      
      if num_valid == 0:
        # Handle edge case with no valid points
        # Create dummy points and indices
        processed_points = np.zeros((self.num_points, points.shape[1]))
        indices = np.zeros(self.num_points, dtype=int)
        return processed_points, indices
          
      if num_valid < self.num_points:
        # Need to augment by sampling
        # Calculate how many additional points we need
        num_needed = self.num_points - num_valid
        
        # Sample with replacement from valid points
        sample_indices = np.random.choice(num_valid, size=num_needed, replace=True)
        
        # Get the sampled points and their original indices
        augmented_points = valid_points[sample_indices]
        augmented_indices = valid_indices[sample_indices]
        
        # Combine original valid points with augmented points
        processed_points = np.concatenate([valid_points, augmented_points], axis=0)
        indices = np.concatenate([valid_indices, augmented_indices])
          
      else:
        # We have more valid points than needed, need to downsample
        if self.point_process_strategy == 'fps':
            # Using Open3D for farthest point sampling
            fps_processor = FPS(valid_points[:, :3], self.num_points)
            _, sampled_indices = fps_processor.fit()
            
            processed_points = valid_points[sampled_indices]
            indices = valid_indices[sampled_indices]

            processed_points = valid_points[sampled_indices]
            indices = valid_indices[sampled_indices]
        else:
            # Random sampling if not using FPS
            sample_indices = np.random.choice(num_valid, size=self.num_points, replace=False)
            processed_points = valid_points[sample_indices]
            indices = valid_indices[sample_indices]
      
      # Ensure we have exactly self.num_points
      assert processed_points.shape[0] == self.num_points
      assert indices.shape[0] == self.num_points
      
      return processed_points, indices

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):

    obs_dict = {}

    if 'state' in self._obs_key:
      obs_dict['state'] = np.concatenate([self.to_np(obs['agent'][k]) for k in obs['agent'].keys()])
    if 'pointcloud' in self._obs_key:
      xyz, idxs = self.process_pcd(obs['pointcloud']['xyzw'])
      if self.use_colored_pcd:
        obs_dict['pointcloud'] = np.concatenate([self.to_np(xyz)[:,:-1], self.to_np(obs['pointcloud']['rgb'][idxs])], axis=-1)
      else:
        obs_dict['pointcloud'] = self.to_np(xyz)[:,:-1]
    if 'rgb' in self._obs_key:
      obs_dict['image'] = self.to_np(obs['sensor_data'][self.cam_name]['rgb'])
    if 'depth' in self._obs_key:
      depth = self.to_np(obs['sensor_data'][self.cam_name]['depth']).astype(np.uint16)
      depth = np.where(depth == 0, self.depth_max_mm, depth) 
      depth = depth.clip(0, self.depth_max_mm)
      obs_dict['image'] = np.concatenate([obs_dict['image'], depth], axis=-1)

    obs_dict = {k: np.asarray(v) for k, v in obs_dict.items()}
    obs_dict.update(
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    return obs_dict

  def render(self):
    image = self._env.render('rgb_array')
    assert image is not None
    return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return elements.Space(np.int32, (), 0, space.n)
    return elements.Space(space.dtype, space.shape, space.low, space.high)


class FPS:
  def __init__(self, pcd_xyz, n_samples):
    self.n_samples = n_samples
    self.pcd_xyz = pcd_xyz
    self.n_pts = pcd_xyz.shape[0]
    self.dim = pcd_xyz.shape[1]
    self.selected_pts = None
    self.selected_pts_expanded = np.zeros(shape=(n_samples, 1, self.dim))
    self.selected_indices = np.zeros(n_samples, dtype=int)  # 선택된 인덱스 추적
    self.remaining_pts = np.copy(pcd_xyz)
    self.original_indices = np.arange(self.n_pts)  # 원본 인덱스 추적

    self.grouping_radius = None
    self.dist_pts_to_selected = None  # Iteratively updated in step(). Finally re-used in group()
    self.labels = None

    # Random pick a start
    self.start_idx = np.random.randint(low=0, high=self.n_pts - 1)
    self.selected_pts_expanded[0] = self.remaining_pts[self.start_idx]
    self.selected_indices[0] = self.original_indices[self.start_idx]  # 시작 인덱스 저장
    self.n_selected_pts = 1

  def get_selected_pts(self):
    self.selected_pts = np.squeeze(self.selected_pts_expanded, axis=1)
    return self.selected_pts
      
  def get_selected_indices(self):
    return self.selected_indices[:self.n_selected_pts]

  def step(self):
    if self.n_selected_pts < self.n_samples:
        self.dist_pts_to_selected = self.__distance__(self.remaining_pts, self.selected_pts_expanded[:self.n_selected_pts]).T
        dist_pts_to_selected_min = np.min(self.dist_pts_to_selected, axis=1, keepdims=True)
        res_selected_idx = np.argmax(dist_pts_to_selected_min)
        self.selected_pts_expanded[self.n_selected_pts] = self.remaining_pts[res_selected_idx]
        self.selected_indices[self.n_selected_pts] = self.original_indices[res_selected_idx]  # 선택된 포인트의 원본 인덱스 저장
        self.n_selected_pts += 1
    else:
        print("Got enough number samples")

  def fit(self):
    for _ in range(1, self.n_samples):
        self.step()
    return self.get_selected_pts(), self.get_selected_indices()  # 포인트와 인덱스 모두 반환

  def group(self, radius):
    self.grouping_radius = radius   # the grouping radius is not actually used
    dists = self.dist_pts_to_selected
    # Ignore the "points"-"selected" relations if it's larger than the radius
    dists = np.where(dists > radius, dists+1000000*radius, dists)
    # Find the relation with the smallest distance.
    # NOTE: the smallest distance may still larger than the radius.
    self.labels = np.argmin(dists, axis=1)
    return self.labels

  @staticmethod
  def __distance__(a, b):
    return np.linalg.norm(a - b, ord=2, axis=2)
