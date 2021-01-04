import collections
from datetime import datetime
from functools import reduce

import gym
import psutil
import torch
from bsuite.utils import gym_wrapper
from gym.wrappers import TransformObservation, FrameStack, TimeAwareObservation
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.algos.dqn.dqn import DQN

from absl import app
from absl import flags

import bsuite
import numpy as np
from rlpyt.utils.logging import logger
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from mlp_dqn_model import MlpDqnModel, MlpDqnAgent
from model_based_runner import WeakAgentModelBasedRunner
from modeled_env_no_latent import ModeledEnv
from run_baseline import ScaledTimeAwareObservation
from shaped_dqn import ShapedDQN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_and_train(bsuite_id, gym_id, run_ID=0,
                    cuda_idx=None, results_dir='./bsuite_shaping', n_parallel=4):
    id = bsuite_id if not gym_id else gym_id
    logger._tf_summary_dir = f'./runs/{id.replace("/", "_")}_{run_ID}_model_shaping_{datetime.now().strftime("%D-%T").replace("/", "_")}'
    logger._tf_summary_writer = SummaryWriter(logger._tf_summary_dir)

    def get_env(*args, **kwargs):
        return GymEnvWrapper(
                TransformObservation(
                    env=FrameStack(
                        num_stack=4,
                        env=(gym_wrapper.GymFromDMEnv(bsuite.load_and_record_to_csv(
                            bsuite_id=bsuite_id,
                            results_dir=results_dir,
                            overwrite=True,
                        )) if not gym_id else gym.make(gym_id))
                    ),
                    f=lambda lazy_frames: np.reshape(np.stack(lazy_frames._frames), -1)
                )
        )

    env_info = get_env()
    obs_ndim = len(env_info.observation_space.shape)
    obs_size = reduce(lambda x, y: x * y, env_info.observation_space.shape)

    def mlp_factory(input_size=obs_size, output_size=env_info.action_space.n, hidden_sizes=None, dueling=False):
        if hidden_sizes is None: hidden_sizes = [64, 64]
        return lambda *args, **kwargs: MlpDqnModel(
            input_size=input_size,
            fc_sizes=hidden_sizes,
            output_size=output_size,
            dueling=dueling,
        )

    latent_step_model = mlp_factory(input_size=obs_size + 1, output_size=obs_size)()
    reward_model = mlp_factory(input_size=2 * obs_size + 1, output_size=1)()
    termination_model = mlp_factory(input_size=2 * obs_size + 1, output_size=1)()

    def get_modeled_env(*args, **kwargs):
        return ModeledEnv(
            latent_step_model=latent_step_model,
            reward_model=reward_model,
            termination_model=termination_model,
            env_cls=get_env,
        )

    model_info = get_modeled_env()

    sampler = SerialSampler(
        EnvCls=get_env,
        batch_T=1,
        batch_B=1,
        env_kwargs=dict(game=bsuite_id),
        eval_env_kwargs=dict(game=bsuite_id),
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )

    weak_sampler = SerialSampler(
        EnvCls=get_modeled_env,
        batch_T=1,
        batch_B=1,
        env_kwargs=dict(game=bsuite_id),
        eval_env_kwargs=dict(game=bsuite_id),
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )


    def shaping(samples):
        # TODO eval/train mode here and in other places
        if logger._iteration <= 1e3:  # FIXME(1e3)
            return 0

        with torch.no_grad():
            obs = (samples.agent_inputs.observation.to(device))  # TODO check if maybe better to keep it on cpu
            obsprim = (samples.target_inputs.observation.to(device))
            qs = weak_agent(obs, samples.agent_inputs.prev_action.to(device), samples.agent_inputs.prev_reward.to(device))
            qsprim = weak_agent(obsprim, samples.target_inputs.prev_action.to(device), samples.target_inputs.prev_reward.to(device))
            vals = 0.995 * torch.max(qsprim, dim=1).values - torch.max(qs, dim=1).values

            if logger._iteration % 1e1 == 0:  # FIXME(1e1)
                with logger.tabular_prefix("Shaping"):
                    logger.record_tabular_misc_stat('ShapedReward', vals.detach().cpu().numpy())
            return vals

    n_steps=3e4
    algo = ShapedDQN(
        # target_update_tau=0.01,
        # target_update_interval=16,
        shaping_function=shaping,
        # target_update_interval=312,
        discount=0.995,
        min_steps_learn=1e3, # 1e3
        eps_steps=n_steps,
        # pri_beta_steps=n_steps,
        double_dqn=True,
        prioritized_replay=True,
        # clip_grad_norm=1,  # FIXME arbitrary
        # n_step_return=4,  # FIXME arbitrary
    )
    weak_algo = DQN(
        # target_update_tau=0.01,
        # target_update_interval=16,
        # target_update_interval=312,
        discount=0.995,
        min_steps_learn=1e3, # 1e3
        eps_steps=n_steps,
        # pri_beta_steps=n_steps,
        double_dqn=True,
        prioritized_replay=True,
        # clip_grad_norm=1,  # FIXME arbitrary
        # n_step_return=4,  # FIXME arbi
    )

    agent = DqnAgent(ModelCls=mlp_factory(hidden_sizes=[512], dueling=False))
    weak_agent = DqnAgent(ModelCls=mlp_factory(input_size=obs_size, hidden_sizes=[512], dueling=False))

    p = psutil.Process()
    runner = WeakAgentModelBasedRunner(
        algo=algo,
        agent=agent,
        sampler=sampler,

        weak_algo=weak_algo,
        weak_agent=weak_agent,
        weak_sampler=weak_sampler,

        env_model=get_modeled_env(),
        # n_steps=num_episodes,
        n_steps=n_steps,  # orig 50e6
        log_interval_steps=1e2,  # orig 1e3
        affinity=dict(cuda_idx=cuda_idx),
        # affinity=dict(cuda_idx=cuda_idx, workers_cpus=p.cpu_affinity()),
    )

    env_info.close()
    model_info.close()
    runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bsuite_id', help='Bsuite id', default='cartpole/0')
    parser.add_argument('--gym_id', help='Gym id', default='CartPole-v0')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0 if torch.cuda.is_available() else None)
    args = parser.parse_args()
    build_and_train(
        bsuite_id=args.bsuite_id,
        gym_id=args.gym_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
