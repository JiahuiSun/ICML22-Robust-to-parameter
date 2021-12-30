import sys
import cloudpickle
import argparse
import time
from os.path import join as pjoin
import os
import numpy as np
import pickle
sys.path.append("/root/jiang21c-supp/MRPO-source-code/MRPOCode_ICML")

from examples.MRPO import base
from examples.baselines.common import set_global_seeds, tf_util as U
from examples.baselines import logger
from examples.util import CircularList, EnvParamDist
from examples.baselines import bench
from examples.baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from examples.baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


class Tester():
    def __init__(
        self,
        env,
        model,
        norm_param,
        test_params,
        n_cpu=20
    ) -> None:
        self.env = env
        self.model = model
        self.test_params = test_params
        self.n_cpu = n_cpu

        self.mean = norm_param['mean']
        self.var = norm_param['var']
        self.clipob = norm_param['clipob']

    def test(self):
        buffer = self.rollout(self.test_params)
        return_list = [data['return'] for param, data in buffer.items()]
        length_list = [data['length'] for param, data in buffer.items()]
        return_list.sort()
        # 指标计算
        logger.logkv('avg_return', np.mean(return_list))
        logger.logkv('wst10_return', np.mean(return_list[:len(return_list)//10]))
        logger.logkv('avg_length', np.mean(length_list))
        logger.dumpkvs()
        self.env.close()
        
    def rollout(self, traj_params=[]):
        # 每个参数收集若干条trajectory
        buffer = {
            tuple(param): {
                'rew': [], 
                'done': []
            } for param in traj_params
        }
        traj_params = CircularList(traj_params)

        # 多进程并行采样，直到所有参数都被采样过
        env_idx_param = {idx: traj_params.pop() for idx in range(self.n_cpu)}
        self.env.set_params(env_idx_param)
        obs = self.env.reset()
        while True:
            obs = np.clip((obs-self.mean) / np.sqrt(self.var), -self.clipob, self.clipob)
            actions, value, state, neglogp = self.model.step(obs)
            obs_next, rewards, dones, infos = self.env.step(actions)
            
            for idx, param in env_idx_param.items():
                buffer[tuple(param)]['rew'].append(rewards[idx])
                buffer[tuple(param)]['done'].append(dones[idx])

            if any(dones):
                env_done_idx = np.where(dones)[0]
                # 采样停止条件：每个param都采样完一条trajectory，可以修改
                traj_params.record([env_idx_param[idx] for idx in env_done_idx])
                if traj_params.is_finish(threshold=self.traj_per_param):
                    break
                env_new_param = {idx: traj_params.pop() for idx in env_done_idx}
                self.env.set_params(env_new_param)
                obs_reset = self.env.reset(env_done_idx)
                obs_next[env_done_idx] = obs_reset
                env_idx_param.update(env_new_param)
            obs = obs_next

        # 数据预处理
        for param, data in buffer.items():
            data['rew'] = np.array(data['rew'])
            data['done'] = np.array(data['done'])
            done_idx = np.where(data['done'])[0]
            data['return'] = np.sum(data['rew'][:max(done_idx)+1]) / len(done_idx)
            data['length'] = (max(done_idx)+1) / len(done_idx)
        return buffer


def main(args):
    # 创建环境
    if args.n_cpu == 1:
        def make_env():
            env = base.make_env(args.env_id, outdir=logger.get_dir())
            env.seed(args.seed)
            env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
            return env
        env = DummyVecEnv([make_env])
    else:
        def make_env(rank):
            def _thunk():
                env = base.make_env(args.env_id, process_idx=rank, outdir=logger.get_dir())
                env.seed(args.seed + rank)
                if logger.get_dir():
                    env = bench.Monitor(env, pjoin(logger.get_dir(), 'train-{}.monitor.json'.format(rank)), allow_early_resets=True)
                return env
            return _thunk
        env = SubprocVecEnv([make_env(i) for i in range(args.n_cpu)])

    # 加载模型和参数
    with open(args.norm_param_path, 'rb') as f:
        norm_param = pickle.load(f)
  
    # 提前生成好的，所有人都一样
    if args.test_params_path:
        test_params = np.load(args.test_params_path)
    else:
        lower_param1, upper_param1, lower_param2, upper_param2 = env.get_lower_upper_bound()
        env_para_dist = EnvParamDist(param_start=[lower_param1, lower_param2], param_end=[upper_param1, upper_param2])
        test_params = env_para_dist.sample(size=(100,))

    with U.make_session(num_cpu=1) as sess:
        with open(args.mk_model_path, 'rb') as fh:
            make_model = cloudpickle.load(fh)
        model = make_model()
        model.load(args.model_path)

        tester = Tester(
            env=env,
            model=model,
            norm_param=norm_param,
            test_params=test_params,
            n_cpu=args.n_cpu
        )
        
        tester.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Distribution Robust RL")
    parser.add_argument('--env_id', type=str, default='SunblazeWalker2d-v0')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--test_params_path', type=str, default='')
    parser.add_argument('--mk_model_path', type=str, default='output/SunblazeWalker2dUniform-v0/DR/20211219_075433/make_model.pkl')
    parser.add_argument('--model_path', type=str, default='output/SunblazeWalker2dUniform-v0/DR/20211219_075433/checkpoints/01000')
    parser.add_argument('--norm_param_path', type=str, default='output/SunblazeWalker2dUniform-v0/DR/20211219_075433/checkpoints/normalize')

    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--traj_per_param', type=int, default=1)
    parser.add_argument('--n_cpu', type=int, default=20)
    parser.add_argument('--test_output', type=str, default='test')	
    args = parser.parse_args()

    logid = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_dir = pjoin(args.test_output, args.env_id, 'MRPO', logid)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger.reset()
    logger.configure(dir=log_dir)
    set_global_seeds(args.seed)

    main(args)
