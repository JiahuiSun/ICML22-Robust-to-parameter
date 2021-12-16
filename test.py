import sys
sys.path.append("/root/jiang21c-supp/MRPO-source-code/MRPOCode_ICML")
from examples.MRPO import base
from examples.baselines.common import set_global_seeds, tf_util as U
from examples.baselines import logger
import time
from os.path import join as pjoin
import os
import numpy as np
import pickle
from examples.baselines import bench
from examples.baselines.common.vec_env.dummy_vec_env import DummyVecEnv


seed = 12
nsteps = 1000
gamma = 0.99
lam = 0.95
paths = 100
eps = 1.0
env_id = 'SunblazeWalker2dRandomNormal-v0'
output = 'output'
logid = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_dir = pjoin(output, 'test', 'MRPO', logid)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logger.reset()
logger.configure(dir=log_dir)

set_global_seeds(seed)
def make_env():
    env = base.make_env(env_id, outdir=logger.get_dir())
    # NOTE: Set the env seed
    env.seed(seed)
    env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
    return env

norm_path = 'output/SunblazeWalker2dRandomNormal-v0/MRPO/20211202_121212/checkpoints/normalize'
with open(norm_path, 'rb') as f:
    obs_norms = pickle.load(f)
clipob = obs_norms['clipob']
mean = obs_norms['mean']
var = obs_norms['var']

with U.make_session(num_cpu=1) as sess:
    pkl_path = 'output/SunblazeWalker2dRandomNormal-v0/MRPO/20211202_121212/make_model.pkl'
    print("Constructing model from " + pkl_path)
    with open(pkl_path, 'rb') as fh:
        import cloudpickle
        make_model = cloudpickle.load(fh)
    model = make_model()
    model_path = 'output/SunblazeWalker2dRandomNormal-v0/MRPO/20211202_121212/checkpoints/01700'
    print("Loading saved model" + model_path)
    model.load(model_path)
    env = DummyVecEnv([make_env])
    # env = VecNormalize(env)
    # runner = MRPORunner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    # obs, returns, masks, actions, values, neglogpacs, states, epinfos, num_episodes, eprewmean_all, epremean_percentile10, \
    #         episodes_returns = runner.run(paths=paths, eps=eps)
    # print(f"avg: {eprewmean_all}, wst: {epremean_percentile10}")

    # Unwrap DummyVecEnv to access mujoco.py object
    env_base = env.envs[0].unwrapped
    # Record a binary success measure if the env supports it
    if hasattr(env_base, 'is_success') and callable(getattr(env_base, 'is_success')):
        success_support = True
    else:
        print("Warning: env does not support binary success, ignoring.")
        success_support = False

    rate = 0.0
    for episode in range(paths):
        obs, state, done = env.reset(), model.initial_state, False
        episode_rew = 0
        success = False
        for step in range(nsteps):
            obs = np.clip((obs-mean) / np.sqrt(var), -clipob, clipob)  # normalize
            action, value, state, _ = model.step(obs, state, np.reshape(np.asarray([done]), (1,)))
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
            if success_support and env_base.is_success():
                #print("success at step {} w/ reward {}".format(step,episode_rew))
                success = True
            if done:
                print("episode {} done at step {} w/ reward {}".format(episode, step, episode_rew))
                break
        rate += success/paths
        # with output_lock:
        #     with open(os.path.join(args.outdir, 'evaluation.json'), 'a') as results_file:
        #         results_file.write(json.dumps({
        #             'reward': episode_rew,
        #             'success': success if success_support else 'N/A',
        #             'environment': env_base.parameters,
        #             'model': args.load,
        #         }, cls=NumpyEncoder))
        #         results_file.write('\n')
