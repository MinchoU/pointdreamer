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
from sapien.core import Pose
from typing import Dict, List

import numpy as np
import sapien.physx as physx
import torch

from mani_skill.render import SAPIEN_RENDER_SYSTEM
from mani_skill.sensors.base_sensor import BaseSensor, BaseSensorConfig
from mani_skill.sensors.camera import Camera
from mani_skill.utils import common

from pointclouds.pcd_utils import apply_pose_to_points

# checked it from self.env.env.unwrapped.segmentation_id_map
SEGMENTATION_FILTER_DICT={
  "PegInsertionSide": [16, 17], # 16: table, 17: ground
  "PickCube"        : [16, 17], # 16: table, 17: ground
}

class FromManiskill(embodied.Env):
  def __init__(self, env, obs_mode='pointcloud+rgb', 
               control_mode=None, num_envs=1, size=(128,128), 
               cam_name='base_camera',
               depth_max_mm=1000,
               obs_frame='base_pose',
               n_downsample_pts=512,
               use_segmented_pts=False,
               ):
    # TODO seeding for reproducibility
    '''
    obs_mode : See https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html
      'pointcloud', 'rgb+depth', 'state_dict', 'state' (flattened), 'sensor_data' 
    control_mode : None will indicate pd_joint_delta_pos in PegInsertionSide-v1, 
      ex) 'pd_joint_delta_pos', 'pd_joint_pos', 'pd_ee_delta_pos', 'pd_ee_delta_pose', 
      'pd_ee_pose', 'pd_joint_target_delta_pos', 'pd_ee_target_delta_pos', 'pd_ee_target_delta_pose', 
      'pd_joint_vel', 'pd_joint_pos_vel', 'pd_joint_delta_pos_vel'
    num_envs : TODO / will parallelization benefit?
    '''
    self.use_pcd = False
    self.use_colored_pcd = False
    self.obs_frame = obs_frame
    self.n_downsample_pts=n_downsample_pts

    self.seg_filters = [value for key, value \
                        in SEGMENTATION_FILTER_DICT.items() \
                        if key in env][0] if use_segmented_pts else []
    
    if 'pointcloud' in obs_mode:
      self.use_pcd = True
      if obs_mode == 'pointcloud+rgb':
        self.use_colored_pcd = True
      obs_mode = 'pointcloud+rgb' # this detours using pointcloud wrapper of maniskill

    self._env = gym.make(env,
                         obs_mode=obs_mode,
                         control_mode=control_mode,
                         num_envs=1,
                         sensor_configs=dict(width=size[0], height=size[1]),
                         )
    self.size = size
    self._env = CPUGymWrapper(self._env)

    self._obs_mode = obs_mode
    # self._env = CPUGymWrapper(self._env) # Make this parallel?

    self.cam_name = cam_name
    self.depth_max_mm = depth_max_mm

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
      pcd_dim = 7 if self.use_colored_pcd else 4
      # spaces["pointcloud"] = elements.Space(np.float32, (self.num_points, pcd_dim))
      # spaces['pointcloud'] = elements.Space(np.float32, (self._env.observation_space['pointcloud']['xyzw'].shape[0], pcd_dim))
      spaces['pointcloud'] = elements.Space(np.float32, (self.size[0]*self.size[1], pcd_dim))
      spaces['render_image'] = elements.Space(np.uint8, (self.size[0], self.size[1], 3))
      spaces['world_pointcloud'] = elements.Space(np.float32, (self.size[0]*self.size[1], 4))
      if self.obs_frame == 'tcp_pose':
        spaces['obs_frame_pose'] = elements.Space(np.float32, (7,))
      # spaces['raw_pointcloud'] = elements.Space(np.float32, (self.size[0]*self.size[1], pcd_dim))

    if 'rgb' in self._obs_key and 'pointcloud' not in self._obs_key:
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
    reward = reward
    terminated = terminated
    truncated = truncated

    self._done = terminated or truncated

    return self._obs(
        obs, reward,
        is_last=bool(truncated),
        is_terminal=bool(terminated))

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):

    obs_dict = {}

    if 'state' in self._obs_key:
      obs_dict['state'] = np.concatenate([obs['agent'][k] for k in obs['agent'].keys()])
    if 'pointcloud' in self._obs_key:
      obs = sensor_data_to_pointcloud(obs, self._env.env.env.env._sensors, "render_image", self.seg_filters)
      obs_dict['world_pointcloud'] = obs['pointcloud']['xyzw']
      xyzw = obs['pointcloud']['xyzw']
      if self.obs_frame == 'tcp_pose': # TODO camera - frame 
        pose = obs['extra'][self.obs_frame]
        if pose.ndim > 1:
          pose = pose[0]
        p, q = pose[:3], pose[3:]
        to_origin = Pose(p=p, q=q).inv()
        xyzw[..., :3] = apply_pose_to_points(xyzw[..., :3], to_origin)
        obs_dict['obs_frame_pose'] = pose

      mask = np.linalg.norm(xyzw[..., :3], axis=-1) > self.depth_max_mm
      xyzw[mask, 3] = 0

      # pad when having insufficienet valid points
      valid_mask = xyzw[:, 3] > 0
      n_valid = int(valid_mask.sum())
      if n_valid < self.n_downsample_pts:
        n_needed = self.n_downsample_pts - n_valid

        valid_idx = np.flatnonzero(valid_mask)
        dup_idx   = np.random.choice(valid_idx, size=n_needed, replace=True)
        fill_pos  = np.flatnonzero(~valid_mask)[:n_needed]
        xyzw[fill_pos] = xyzw[dup_idx]

      if self.use_colored_pcd:
        obs_dict['pointcloud'] = np.concatenate([xyzw, obs['pointcloud']['rgb']], axis=-1)
      else:
        obs_dict['pointcloud'] = xyzw
      

    if 'pointcloud' not in self._obs_key and 'rgb' in self._obs_key:
      obs_dict['image'] = obs['sensor_data'][self.cam_name]['rgb']

    if 'render_image' in obs:
      obs_dict['render_image'] = obs['render_image']

    if 'depth' in self._obs_key:
      depth = obs['sensor_data'][self.cam_name]['depth'].astype(np.uint16)
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

def sensor_data_to_pointcloud(observation: Dict, sensors: Dict[str, BaseSensor], img_key: str, seg_filters: List[int]):
    """convert all camera data in sensor to pointcloud data"""
    sensor_data = observation["sensor_data"]
    camera_params = observation["sensor_param"]
    pointcloud_obs = dict()

    for (cam_uid, images), (sensor_uid, sensor) in zip(
        sensor_data.items(), sensors.items()
    ):
        assert cam_uid == sensor_uid
        if isinstance(sensor, Camera):
            cam_pcd = {}
            # TODO: double check if the .clone()s are necessary
            # Each pixel is (x, y, z, actor_id) in OpenGL camera space
            # actor_id = 0 for the background
            images: Dict[str, np.ndarray]
            position = images["position"].copy().astype(np.float32)
            segmentation = images["segmentation"].copy()
            w = (segmentation != 0) & (~np.isin(segmentation, seg_filters))
            position[..., :3] /= 1000.0

            # Convert to world space
            cam2world = camera_params[cam_uid]["cam2world_gl"]
            xyzw = np.concatenate([position, w], axis=-1).reshape(
                -1, 4
            ) @ cam2world.transpose(1, 0)
            cam_pcd["xyzw"] = xyzw

            # Extra keys
            if "rgb" in images:
                rgb = images["rgb"][..., :3].copy()
                cam_pcd["rgb"] = rgb.reshape(-1, 3)
            if "segmentation" in images:
                cam_pcd["segmentation"] = segmentation.reshape(
                    -1, 1
                )

            pointcloud_obs[cam_uid] = cam_pcd
    if len(pointcloud_obs.keys())>1:
      observation[img_key] = np.concatenate([observation["sensor_data"][k]["rgb"] for k in pointcloud_obs.keys()], axis=0)
    else:
      observation[img_key] = observation["sensor_data"][list(pointcloud_obs.keys())[0]]["rgb"]
    for k in pointcloud_obs.keys():
        del observation["sensor_data"][k]
    pointcloud_obs = common.merge_dicts(pointcloud_obs.values())
    for key, value in pointcloud_obs.items():
        pointcloud_obs[key] = np.concatenate(value, axis=1)
    observation["pointcloud"] = pointcloud_obs

    return observation
