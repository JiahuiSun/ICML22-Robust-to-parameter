import time
from os.path import join as pjoin
import argparse
import multiprocessing
import os
from gym.utils.seeding import create_seed
import tensorflow as tf
import sys
sys.path.append("/root/jiang21c-supp/MRPO-source-code/MRPOCode_ICML")

from examples.baselines import logger
from examples.baselines import bench
from examples.epopt import base
from examples.epopt import epopt_ppo2
from examples.epopt import epopt_a2c
from examples.baselines.common import set_global_seeds
from examples.baselines.common.vec_env.vec_normalize import VecNormalize
from examples.baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from examples.baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def train_epopt(
        env_id,
        total_episodes,
        seed,
        lr,
        epsilon,
        activate_at,
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
            # Set the env seed
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir())
            return env
        env = DummyVecEnv([make_env])
    else:
        def make_env(rank):
            def _thunk():
                env = base.make_env(env_id, process_idx=rank, outdir=logger.get_dir())
                env.seed(seed + rank)
                if logger.get_dir():
                    env = bench.Monitor(env, pjoin(logger.get_dir(), 'train-{}.monitor.json'.format(rank)), allow_early_resets=True)
                return env
            return _thunk
        env = SubprocVecEnv([make_env(i) for i in range(ncpu)])
    env = VecNormalize(env)
    set_global_seeds(seed)

    # NOTE: takes iterations as arg, not frac. like other provided callables
    def adaptive_epsilon_fn(epsilon_const, fix_steps, init_value=1.0):
        """Fix epsilon to <init_value> for the first <fix_steps> iterations"""
        return (lambda i: epsilon_const if i > fix_steps else init_value)

    if policy=='mlp':
        policy_fn = base.mlp_policy
    else:
        raise NotImplementedError

    if algorithm == 'ppo2':
        if 'Breakout' in env_id or 'SpaceInvaders' in env_id:
            raise NotImplementedError
        else:
            print("Running ppo2 with mujoco/roboschool settings")
            epopt_ppo2.learn(
                policy=policy_fn,
                env=env,
                nminibatches=nminibatches,
                lam=0.95,
                gamma=0.99,
                noptepochs=10,
                log_interval=10,
                ent_coef=0.0,
                lr=lr,
                cliprange=0.2,
                total_episodes=total_episodes,
                paths=paths,
                epsilon=adaptive_epsilon_fn(epsilon, activate_at),
                save_interval=10,
            )
    elif algorithm == 'a2c':
        if 'Breakout' in env_id or 'SpaceInvaders' in env_id:
            raise NotImplementedError
        else:
            epopt_a2c.learn(
                policy=policy_fn,
                env=env,
                nsteps=nsteps,
                #nsteps=5,
                total_episodes=total_episodes,
                max_timesteps=total_episodes*env.venv.envs[0].spec.max_episode_steps,
                #lr=7e-4,
                lr=lr,
                ent_coef=ent_coef,  # default 0.01 in baselines, 0.0001 in chainer A3C
                lrschedule='linear',
                epsilon=1e-5,  # NOT the EPOpt epsilon
                alpha=0.99,
                gamma=0.99,
                # EPOpt specific
                paths=paths,
                epopt_epsilon=adaptive_epsilon_fn(epsilon, activate_at),
                # functions the same as old pposgd checkpointer
                log_interval=1,
                save_interval=100,
            )
    # some other inner RL algorithm
    else:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='SunblazeWalker2dUniform-v0')
    parser.add_argument('--baseline', type=str, choices=['DR', 'EPOpt' ], default='DR')
    parser.add_argument('--seed', type=int, default=12, help='RNG seed, defaults to random')
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--processes', default=4, help='int or "max" for all')

    # EPOpt specific
    parser.add_argument('--epsilon', type=float, default=1.0)
    # EPOpt paper keept epsilon=1 until iters>100 (max 200 iters)
    parser.add_argument('--activate', type=int, default=10, help='How long to fix epsilon to 1.0 before e')
    parser.add_argument('--paths', type=int, default=100, help='number of trajectories to sample from each iteration')
    parser.add_argument('--algorithm', type=str, choices=['ppo2', 'a2c'],
        default='ppo2', help='Inner batch policy optimization algorithm')
    parser.add_argument('--policy', choices=['mlp', 'lstm'],
        default='mlp', help='Policy architecture')
    parser.add_argument('--total-episodes', type=int, default=5e4)

    # RL algo. yyperparameters
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--nsteps', type=int, default=2048)
    parser.add_argument('--ent-coef', type=float, default=1e-2, help='Only relevant for A2C')
    parser.add_argument('--nminibatches', type=int, default=128, help='Only relevant for PPO2')
    args = parser.parse_args()

    # gpu config
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.cuda}"
    # Configure logger
    logid = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_dir = pjoin(args.output, args.env, f'{args.baseline}', logid)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        logger.reset()
        logger.configure(dir=log_dir)

    # If seed is unspecified, generate a pseudorandom one
    if not args.seed:
        # "Seed must be between 0 and 2**32 - 1"
        seed = create_seed(args.seed, max_bytes=4)
    else:
        seed = args.seed
    with open(os.path.join(log_dir, 'args.txt'), 'w') as fout:
        fout.write(f"{args}")

    if args.processes == 'max':
        ncpu = multiprocessing.cpu_count()
        # from: https://github.com/openai/baselines/blob/1f8a03f3a62367526f20215188fb5ea4b9ec27e0/baselines/ppo2/run_atari.py#L15
        if sys.platform == 'darwin': ncpu //= 2
    else:
        try:
            ncpu = int(args.processes)
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid number of processes")

    train_epopt(
        args.env,
        total_episodes=args.total_episodes,
        seed=seed,
        lr=args.lr,
        epsilon=args.epsilon,
        activate_at=args.activate,
        paths=args.paths,
        algorithm=args.algorithm,
        policy=args.policy,
        ncpu=ncpu,
        nsteps=args.nsteps,
        nminibatches=args.nminibatches,
        ent_coef=args.ent_coef,  # default 0.01 in baselines, 0.0001 in chainer A3C
    )


if __name__ == '__main__':
    main()
