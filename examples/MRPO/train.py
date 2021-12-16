import argparse
import multiprocessing
import os
from os.path import join as pjoin
import time
from gym.utils.seeding import create_seed
import tensorflow as tf
import sys
sys.path.append("/root/jiang21c-supp/MRPO-source-code/MRPOCode_ICML")

from examples.baselines import logger
from examples.baselines import bench
from examples.MRPO import base
from examples.MRPO import MRPO_ppo2
from examples.baselines.common import set_global_seeds
from examples.baselines.common.vec_env.vec_normalize import VecNormalize
from examples.baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from examples.baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def train_MRPO(
        env_id,
        total_episodes,
        seed,
        lr,
        paths,
        algorithm,
        policy,
        ncpu,
        nsteps,
        nminibatches,
        ent_coef,
        ):
    # Set up environment
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    if ncpu == 1:
        def make_env():
            env = base.make_env(env_id, outdir=logger.get_dir())
            # NOTE: Set the env seed
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
            return env
        env = DummyVecEnv([make_env])
    else:
        def make_env(rank):
            def _thunk():
                env = base.make_env(env_id, process_idx=rank, outdir=logger.get_dir())
                env.seed(seed + rank)

                if logger.get_dir():
                    env = bench.Monitor(env, os.path.join(logger.get_dir(), 'train-{}.monitor.json'.format(rank)))

                return env
            return _thunk
        env = SubprocVecEnv([make_env(i) for i in range(ncpu)])
    env = VecNormalize(env)
    set_global_seeds(seed)

    # NOTE: takes iterations as arg, not frac. like other provided callables
    if policy=='mlp':
        policy_fn = base.mlp_policy
    else:
        raise NotImplementedError

    if algorithm == 'MRPO':
        if 'Breakout' in env_id or 'SpaceInvaders' in env_id:
            raise NotImplementedError
        else:
            print("Running ppo2 with mujoco/roboschool settings")
            MRPO_ppo2.learn(
                policy=policy_fn,
                env=env,
                env_id=env_id,
                nsteps=nsteps,
                nminibatches=nminibatches,
                #nsteps=2048,  # used by ppo2_baselines / original paper
                #nsteps=1024,  # used by ppo_baselines
                #nminibatches=32,  # used by ppo2_baselines
                #nminibatches=64,  # used by ppo_baselines
                lam=0.95,
                gamma=0.99,
                noptepochs=10,
                log_interval=1,
                ent_coef=0.0,
                lr=lr,
                #lr=3e-4,
                cliprange=0.2,
                #total_timesteps=num_timesteps,
                total_episodes=total_episodes,
                # EPOpt specific
                paths=paths,
                # functions the same as old pposgd checkpointer
                save_interval=100
            )
    # some other inner RL algorithm
    else:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='SunblazeWalker2dRandomNormal-v0')
    parser.add_argument('--seed', type=int, default=6, help='RNG seed, defaults to random')
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--cuda', type=int, default=-1)
    # TODO: 跑多核有问题
    parser.add_argument('--processes', default=1, help='int or "max" for all')
   
    # MRPO sepcific
    parser.add_argument('--eps-start', type=float, default=1.0)
    parser.add_argument('--eps-end', type=float, default=40)
    parser.add_argument('--eps-raise', type=float, default=1.005)

    parser.add_argument('--paths', type=int, default=50, help='number of trajectories to sample from each iteration')
    parser.add_argument('--algorithm', type=str, choices=['ppo2', 'a2c', 'MRPO' ],
        default='MRPO', help='Inner batch policy optimization algorithm')
    parser.add_argument('--policy', choices=['mlp', 'lstm'],
        default='mlp', help='Policy architecture')

    # Episode-modification specific:
    parser.add_argument('--total-episodes', type=int, default=2.5e4) # 5e4
    # RL algo. hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--nsteps', type=int, default=2048)
    parser.add_argument('--ent-coef', type=float, default=1e-2, help='Only relevant for A2C')
    parser.add_argument('--nminibatches', type=int, default=32, help='Only relevant for PPO2')
    args = parser.parse_args()

    # gpu config
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.cuda}"

    # Configure logger
    logid = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_dir = pjoin(args.output, args.env, 'MRPO', logid)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger.reset()
    logger.configure(dir=log_dir)

    # If seed is unspecified, generate a pseudorandom one
    if not args.seed:
        seed = create_seed(args.seed, max_bytes=4)
    else:
        seed = args.seed 
    with open(os.path.join(log_dir, 'args.txt'), 'w') as fout:
        fout.write(f"{args}")

    if args.processes == 'max':
        ncpu = multiprocessing.cpu_count()
        if sys.platform == 'darwin': ncpu //= 2
    else:
        try:
            ncpu = int(args.processes)
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid number of processes")

    train_MRPO(
        args.env,
        total_episodes=args.total_episodes,
        seed=seed,
        lr=args.lr,
        paths=args.paths,
        algorithm=args.algorithm,
        policy=args.policy,
        ncpu=ncpu,
        nsteps=args.nsteps,
        nminibatches=args.nminibatches,
        ent_coef=args.ent_coef  # default 0.01 in baselines, 0.0001 in chainer A3C
        )


if __name__ == '__main__':
    main()
