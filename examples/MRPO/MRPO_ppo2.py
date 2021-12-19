import os
import time
import joblib
import numpy as np
import os.path as osp
import pickle
import tensorflow as tf
from tensorboardX import SummaryWriter

from examples.baselines import logger
from examples.baselines.common import explained_variance
from examples.ppo2_baselines.ppo2_episodes import constfn


class MRPOModel(object):
    def __init__(self, policy, ob_space, ac_space, nbatch_act, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        # begin diff
        train_model = policy(sess, ob_space, ac_space, nbatch_act, None, reuse=True)
        # end diff

        A = train_model.pdtype.sample_placeholder([None]) # action
        ADV = tf.placeholder(tf.float32, [None])  # adavantage
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])
        neglogpac = train_model.pd.neglogp(A)  # old policy

        # entropy bonus
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # value loss
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # clipped surrogate objective
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef   # L^{CLIP}, entropy bonus, L^{VF}

        # train the model, separate gradients computation and update enables operations to gradients, like clip
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange,  obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values  # returns is the estimator for Q
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)


class MRPORunner():
    """Modified the trajectory generator in PPO2 to follow EPOpt-e"""
    def __init__(self, env, model, gamma, lam):
        self.env = env
        self.model = model
        self.gamma = gamma
        self.lam = lam
        self.n_parallel = env.num_envs
        self.states = None
        
    def run(self, paths, eps):
        # 收集paths条数据
        n_mb_obs = [[] for _ in range(paths)]
        n_mb_rewards = [[] for _ in range(paths)]
        n_mb_actions = [[] for _ in range(paths)]
        n_mb_values = [[] for _ in range(paths)]
        n_mb_dones = [[] for _ in range(paths)]
        n_mb_neglogpacs = [[] for _ in range(paths)]
        n_mb_envparam = [[] for _ in range(paths)]
        n_Jpi = np.zeros(paths, dtype=np.float)
        n_Jlen = np.zeros(paths, dtype=np.float)

        path_cnt = 0
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_Jpi, mb_Jlen, mb_params = [], [], []
        obs, _  = self.env.reset()
        while True:
            actions, values, _, neglogpacs = self.model.step(obs)
            mb_obs.append(obs)
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_params.append(self.env.get_params())

            obs, rewards, dones, infos, _ = self.env.step(actions)
            mb_rewards.append(rewards)
            mb_dones.append(dones)
            Jpi, Jlen = [], []
            for info in infos:
                if info.get('episode'):
                    Jpi.append(info['episode']['r'])
                    Jlen.append(info['episode']['l'])
                else:
                    Jpi.append(0)
                    Jlen.append(0)
            mb_Jpi.append(Jpi)
            mb_Jlen.append(Jlen)

            if any(dones):
                path_cnt += sum(dones)
                if path_cnt >= paths:
                    break
        
        # 把trajectory从4条轨道中分离到100条轨道
        mb_obs = np.array(mb_obs).transpose(1, 0, 2)
        mb_actions = np.array(mb_actions).transpose(1, 0, 2)
        mb_rewards = np.array(mb_rewards).transpose(1, 0)
        mb_values = np.array(mb_values).transpose(1, 0)
        mb_dones = np.array(mb_dones).transpose(1, 0)
        mb_neglogpacs = np.array(mb_neglogpacs).transpose(1, 0)
        mb_Jpi = np.array(mb_Jpi).transpose(1, 0)
        mb_Jlen = np.array(mb_Jlen).transpose(1, 0)
        mb_params = np.array(mb_params).transpose(1, 0, 2)

        N = 0
        exit_flag = False
        for i in range(self.n_parallel):
            pre_inds = np.where(mb_dones[i])[0]
            inds = [0]
            inds.extend(pre_inds+1)
            for st, end in zip(inds[0:-1], inds[1:]):
                n_mb_obs[N] = mb_obs[i, st:end]
                n_mb_actions[N] = mb_actions[i, st:end]
                n_mb_rewards[N] = mb_rewards[i, st:end]
                n_mb_values[N] = mb_values[i, st:end]
                n_mb_dones[N] = mb_dones[i, st:end]
                n_mb_neglogpacs[N] = mb_neglogpacs[i, st:end]
                n_mb_envparam[N] = mb_params[i, end-1]
                n_Jpi[N] = mb_Jpi[i, end-1]
                n_Jlen[N] = mb_Jlen[i, end-1]
                N += 1
                if N >= paths:
                    exit_flag = True
                    break
            if exit_flag:
                break

        # Compute the worst epsilon paths and concatenate them
        episode_returns = np.array([sum(r) for r in n_mb_rewards]).squeeze()
        cutoff = np.min(episode_returns)
        minidx_ = np.argwhere(episode_returns == cutoff).squeeze()
        if isinstance(minidx_, list):
            print('list')
            minidx = minidx_[0]
        else:
            minidx = minidx_
        wst_envparam = n_mb_envparam[minidx]

        # Trajectory selection
        num_episodes = 0
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        lower_param1, upper_param1, lower_param2, upper_param2 = self.env.get_lower_upper_bound()
        envparam_normalizatio = np.zeros(shape=np.array(wst_envparam).shape)
        envparam_normalizatio[0] = (upper_param1 - lower_param1)*2
        envparam_normalizatio[1] = (upper_param2 - lower_param2) # *2can be tuned 

        return_cutoff = np.percentile(episode_returns, 10)
        episode_returns_percen10 = []
        wst_percen_params_list = []
        wst_percen_idx_list = []
        # add worst 10% envs
        for N in range(paths):
            if episode_returns[N] <= return_cutoff:
                episode_returns_percen10.append(episode_returns[N])
                wst_percen_params_list.append(n_mb_envparam[N])
                wst_percen_idx_list.append(N)
                num_episodes += 1

                # concatenate
                mb_obs.extend(n_mb_obs[N])
                mb_rewards.extend(n_mb_rewards[N])
                mb_actions.extend(n_mb_actions[N])
                mb_values.extend(n_mb_values[N])
                mb_dones.extend(n_mb_dones[N])
                mb_neglogpacs.extend(n_mb_neglogpacs[N])

        # add trajectories that is chosen by their metric eps
        wst_percen_params_arr = np.array(wst_percen_params_list)
        for N in range(paths):
            envparam_comp_arr = (wst_percen_params_arr -  np.array(n_mb_envparam[N]))/ np.array(envparam_normalizatio)
            envparam_comp = np.sum(envparam_comp_arr, axis=1).__abs__()
            # 出现了，这这就是line9计算的值，收益 - eps * 参数差距
            objs = episode_returns[N] - eps * envparam_comp
            if all(objs>episode_returns_percen10) and N not in wst_percen_idx_list:
                num_episodes += 1

                # concatenate
                mb_obs.extend(n_mb_obs[N])
                mb_rewards.extend(n_mb_rewards[N])
                mb_actions.extend(n_mb_actions[N])
                mb_values.extend(n_mb_values[N])
                mb_dones.extend(n_mb_dones[N])
                mb_neglogpacs.extend(n_mb_neglogpacs[N])

        total_steps = len(mb_rewards)  # 所有trajectory一共有多少step
        ratio = paths // 10
        epremean_percentile10 = np.mean(np.partition(n_Jpi, ratio)[:ratio])
        eprewmean_all = np.mean(n_Jpi)
        avg_traj_len = np.mean(n_Jlen)

        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        # GAE
        last_obs = mb_obs[-self.n_parallel:]
        last_done = mb_dones[-1]
        last_values = self.model.value(last_obs)[-1]  # value function
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(total_steps)):
            if t == total_steps - 1:
                nextnonterminal = 1.0 - last_done
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, self.states, num_episodes, eprewmean_all, epremean_percentile10, avg_traj_len


def learn(policy, env, total_episodes=5e4, lr=3e-4,
        ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, 
        gamma=0.99, lam=0.95, log_interval=10, 
        nminibatches=64, noptepochs=4, cliprange=0.2,
        save_interval=0, keep_all_ckpt=False, paths=100, 
        eps_start=1.0, eps_end=40, eps_raise=1.005
    ):
    # Callable lr and cliprange don't work (at the moment) with the
    # total_episodes terminating condition
    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        raise NotImplementedError
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        raise NotImplementedError

    total_episodes = int(total_episodes)
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    eps = eps_start

    make_model = lambda : MRPOModel(policy=policy, ob_space=ob_space, 
        ac_space=ac_space, nbatch_act=nenvs, ent_coef=ent_coef, 
        vf_coef=vf_coef, max_grad_norm=max_grad_norm)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))

    model = make_model()
    runner = MRPORunner(env=env, model=model, gamma=gamma, lam=lam)

    tfirststart = time.time()
    update = 0
    episodes_so_far = 0
    old_savepath = None
    eprewmean_all_arr = []
    epremean_percentile10_arr = []
    writer = SummaryWriter(logger.get_dir())
    while True:
        update += 1
        if episodes_so_far > total_episodes:
            break

        frac = 1.0 - (update - 1.0) / total_episodes
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        obs, returns, masks, actions, values, neglogpacs, states, num_episodes, eprewmean_all, epremean_percentile10, avg_traj_len = runner.run(paths=paths, eps=eps)
        assert num_episodes==np.sum(masks), (num_episodes, np.sum(masks))
        episodes_so_far += num_episodes

        # if num_episodes>30:
        #     eps = eps_raise * eps
        eps = eps_raise * eps
        # eps = min(eps_end, eps_raise * eps)

        eprewmean_all_arr.append(eprewmean_all)
        epremean_percentile10_arr.append(epremean_percentile10)
        writer.add_scalar('wst10', epremean_percentile10, update)
        writer.add_scalar('avg', eprewmean_all, update)

        mblossvals = []
        if states is None: # nonrecurrent version
            if returns.shape[0] > nminibatches and nminibatches > 0:
                n_batch = returns.shape[0] // nminibatches
                inds = np.arange(returns.shape[0])
                for _ in range(noptepochs):
                    np.random.shuffle(inds)
                    for i in range(n_batch):
                        if i+1 == n_batch:
                            mbinds = inds[i*nminibatches:]
                        else:
                            mbinds = inds[i*nminibatches:(i+1)*nminibatches]
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices))
            else:
                for _ in range(noptepochs):
                    mblossvals.append(model.train(lrnow, cliprangenow, *(obs, returns, masks, actions, values, neglogpacs)))
        else: # recurrent version
            raise NotImplementedError("Use examples.epopt_lstm")

        lossvals = np.mean(mblossvals, axis=0)
        # log
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv('time_elapsed', time.time() - tfirststart)
            logger.logkv("epoch", update)
            logger.logkv("episodes_so_far", episodes_so_far)
            logger.logkv("envs choosed", num_episodes)
            logger.logkv("eps", eps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eplenmean', avg_traj_len)
            logger.logkv('avg_return', eprewmean_all)
            logger.logkv('wst10_return', epremean_percentile10)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()

        # save model
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
            obs_norms = {}
            obs_norms['clipob'] = env.clipob
            obs_norms['mean'] = env.ob_rms.mean
            obs_norms['var'] = env.ob_rms.var+env.epsilon
            with open(osp.join(checkdir, 'normalize'), 'wb') as f:
                pickle.dump(obs_norms, f, pickle.HIGHEST_PROTOCOL)

            if not keep_all_ckpt and old_savepath:
                print('Removing previous checkpoint', old_savepath)
                os.remove(old_savepath)
            old_savepath = savepath
    savepath = osp.join(checkdir, 'final')
    model.save(savepath)
    obs_norms = {}
    obs_norms['clipob'] = env.clipob
    obs_norms['mean'] = env.ob_rms.mean
    obs_norms['var'] = env.ob_rms.var+env.epsilon
    with open(osp.join(checkdir, 'normalize_final'), 'wb') as f:
        pickle.dump(obs_norms, f, pickle.HIGHEST_PROTOCOL)
    env.close()
    # save data
    logdir = logger.get_dir()
    filename1 = f'{logdir}/avg.txt'
    np.savetxt(filename1, eprewmean_all_arr)
    filename2 = f'{logdir}/wst10.txt'
    np.savetxt(filename2, epremean_percentile10_arr)
