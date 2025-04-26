import functools

import elements
import embodied
import gymnasium as gym
import mani_skill
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
import numpy as np


class FromManiskill(embodied.Env):
  def __init__(self, env, obs_mode='pointcloud', 
               control_mode=None, num_envs=1, size=(128,128), 
               cam_name='base_camera',
               depth_max_mm=1000,):
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
    self._env = gym.make(env,
                         obs_mode=obs_mode,
                         control_mode=control_mode,
                         num_envs=num_envs,
                         sensor_configs=dict(width=size[0], height=size[1]),
                         
                         )
    self._obs_mode = obs_mode
    self._env = CPUGymWrapper(self._env)

    self.cam_name = cam_name
    self.depth_max_mm = depth_max_mm

    if type(self.cam_name) != str and "rgb" in obs_mode:
      raise ValueError("Only single camera is supported for RGB observation.")
      # for pointclouds, data is automatically gained data using all cameras
      # , so need to modify default setup
    self._obs_key = obs_mode
    # if self._obs_mode == 'pointcloud':
    #   self._obs_key = 'pointcloud'
    # elif self._obs_mode == 'rgb+depth':
    #   self._obs_key = ['image', 'depth']
    # elif self._obs_mode == 'rgb':
    #   self._obs_key = 'image'
    # elif self._obs_mode == 'state':
    #   self._obs_key = 'state'
    # elif self._obs_mode == 'rgb+state':
    #   self._obs_key = ['image', 'state']
    # elif self._obs_mode == 'pointcloud+state':
    #   self._obs_key = ['pointcloud', 'state']
    
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
      spaces['pointcloud'] = elements.Space(np.float32, (self._env.observation_space['pointcloud'].shape[0], 3))
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
      obs_dict['pointcloud'] = obs['pointcloud']
    if 'rgb' in self._obs_key:
      obs_dict['image'] = obs['sensor_data'][self.cam_name]['rgb']
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
