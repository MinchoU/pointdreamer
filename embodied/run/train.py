import collections
from functools import partial as bind

import elements
import embodied
import numpy as np
from pointclouds.pcd_utils import viz_world_pcd_img, viz_pcd_img
def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif value.dtype == np.uint16 and value.ndim == 3 and value.shape[-1] == 4:
        if worker == 0:
          episode.add(f'policy_{key}', value[...,:3].astype(np.uint8), agg='stack')
      elif key == "pointcloud":
        assert "raw_pointcloud" in tran
        if worker == 0:
          w, h, pad = 256, 256, 4
          camera_settings = {
                              'camera_position': (1.0038888282878466, -0.6127577168023789, 0.5753893737567826),
                              'camera_focal_point': (-0.044459663331508636, 0.021982893347740173, -0.07509130239486694),
                              'camera_up': (-0.4032507264417322, 0.23916935296929345, 0.8832818758609648),
                            }
          # pcd_img = viz_pcd_img(value, width=512, height=512, **camera_settings)
          if "obs_frame_pose" in tran:
            pcd_img = viz_world_pcd_img(value, tran["obs_frame_pose"], width=w, height=h, **camera_settings)
            raw_pcd_img = viz_world_pcd_img(tran["raw_pointcloud"], tran["obs_frame_pose"], width=w, height=h, **camera_settings)
          else:
            pcd_img = viz_pcd_img(value, width=w, height=h, **camera_settings)
            raw_pcd_img = viz_pcd_img(tran["raw_pointcloud"], width=w, height=h, **camera_settings)
          # raw_pcd_img = viz_pcd_img(tran["raw_pointcloud"], width=512, height=512,**camera_settings)
          vertical_padding = 255*np.ones((h,pad,3), dtype=np.uint8)
          pcd_img_cat = np.concatenate([raw_pcd_img, vertical_padding, pcd_img], axis=1)
        episode.add(f'policy_{key}', pcd_img_cat, agg='stack')

      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(logfn)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  if agent.model.use_pcd:
    downsampler = lambda *args: agent.downsample(*args)
  else:
    downsampler = None

  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, downsampler, steps=10)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats(log=True)['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()