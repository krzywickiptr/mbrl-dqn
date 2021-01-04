
"""
Runs one instance of the Atari environment and optimizes using DQN algorithm.
Can use a GPU for the agent (applies to both sample and train). No parallelism
employed, so everything happens in one python process; can be easier to debug.

The kwarg snapshot_mode="last" to logger context will save the latest model at
every log point (see inside the logger for other options).

In viskit, whatever (nested) key-value pairs appear in config will become plottable
keys for showing several experiments.  If you need to add more after an experiment,
use rlpyt.utils.logging.context.add_exp_param().

"""
from datetime import datetime

import bsuite
import gym
import psutil
import torch
from bsuite.utils import gym_wrapper
from gym import ObservationWrapper
from gym.spaces import Box
from gym.wrappers import FrameStack, TransformObservation, TimeAwareObservation
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging import logger

from mlp_dqn_model import MlpDqnModel, MlpDqnAgent
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.runners.minibatch_rl import MinibatchRlEval

from rlpyt.envs.gym import GymEnvWrapper
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class ScaledTimeAwareObservation(TimeAwareObservation):

    def __init__(self, env, scale):
        super(ScaledTimeAwareObservation, self).__init__(env)
        self.scale = scale

    def observation(self, observation):
        return np.append(observation, self.t // self.scale)


def build_and_train(bsuite_id, gym_id, run_ID=0, cuda_idx=None,
                    results_dir='./bsuite_baseline', n_parallel=8):
    id = bsuite_id if not gym_id else gym_id
    logger._tf_summary_dir = f'./runs/{id.replace("/", "_")}_{run_ID}_baseline_{datetime.now().strftime("%D-%T").replace("/", "_")}'
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

    sampler = SerialSampler(  # TODO (Async)GpuSampler
        EnvCls=get_env,
        env_kwargs=dict(game=bsuite_id),
        eval_env_kwargs=dict(game=bsuite_id),
        batch_T=1,  # Four time-steps per sampler iteration. (Only influence count)
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )

    n_steps=3e4
    algo = DQN(
        discount=0.995,
        min_steps_learn=1e3,
        eps_steps=n_steps,
        # delta_clip=None,
        # learning_rate=1e-4,
        # target_update_tau=500,
        # target_update_tau=0.01,
        # target_update_interval=100,
        double_dqn=True,
        prioritized_replay=True,
        # clip_grad_norm=1,  # FIXME arbitrary
        # n_step_return=2,  # FIXME arbitrary
        # clip_grad_norm=1000000,
    )  # Run with defaults.
    # agent = MlpDqnAgent(ModelCls=lambda *args, **kwargs: MlpDqnModel(*args, **kwargs, dueling=True))
    agent = MlpDqnAgent(ModelCls=MlpDqnModel)

    p = psutil.Process()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=n_steps,  # orig 50e6
        log_interval_steps=1e2, # orig 1e3,
        affinity=dict(cuda_idx=cuda_idx),
        # affinity=dict(cuda_idx=cuda_idx, workers_cpus=p.cpu_affinity()[:n_parallel]),
    )
    runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
