from statistics import mean

import torch
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging import logger
import time

import psutil
from rlpyt.utils.seed import set_seed, make_seed

from modeled_env import ModeledEnv


class WeakAgentModelBasedRunner(MinibatchRlEval):
    """
    Runs RL on minibatches; tracks performance offline using evaluation
    trajectories.
    """

    def __init__(self,
                 env_model: ModeledEnv,
                 algo, agent, sampler, weak_algo, weak_agent, weak_sampler,
                 n_steps, seed=None, affinity=None, log_interval_steps=1e5):
        self.env_model = env_model
        self.weak_algo = weak_algo
        self.weak_agent = weak_agent
        self.weak_sampler = weak_sampler
        super().__init__(algo, agent, sampler, n_steps, seed, affinity, log_interval_steps)

    def startup(self):
        """
        Sets hardware affinities, initializes the following: 1) sampler (which
        should initialize the agent), 2) agent device and data-parallel wrapper (if applicable),
        3) algorithm, 4) logger.
        """
        p = psutil.Process()
        try:
            if (self.affinity.get("master_cpus", None) is not None and
                    self.affinity.get("set_affinity", True)):
                p.cpu_affinity(self.affinity["master_cpus"])
            cpu_affin = p.cpu_affinity()
        except AttributeError:
            cpu_affin = "UNAVAILABLE MacOS"
        logger.log(f"Runner {getattr(self, 'rank', '')} master CPU affinity: "
            f"{cpu_affin}.")
        if self.affinity.get("master_torch_threads", None) is not None:
            torch.set_num_threads(self.affinity["master_torch_threads"])
        logger.log(f"Runner {getattr(self, 'rank', '')} master Torch threads: "
            f"{torch.get_num_threads()}.")
        if self.seed is None:
            self.seed = make_seed()
        set_seed(self.seed)
        self.rank = rank = getattr(self, "rank", 0)
        self.world_size = world_size = getattr(self, "world_size", 1)
        examples = self.sampler.initialize(
            agent=self.agent,  # Agent gets initialized in sampler.
            affinity=self.affinity,
            seed=self.seed + 1,
            bootstrap_value=getattr(self.algo, "bootstrap_value", False),
            traj_info_kwargs=self.get_traj_info_kwargs(),
            rank=rank,
            world_size=world_size,
        )
        weak_examples = self.weak_sampler.initialize(
            agent=self.weak_agent,  # Agent gets initialized in sampler.
            affinity=self.affinity,
            seed=self.seed + 1,
            bootstrap_value=getattr(self.weak_algo, "bootstrap_value", False),
            traj_info_kwargs=self.get_traj_info_kwargs(),
            rank=rank,
            world_size=world_size,
        )
        self.itr_batch_size = self.sampler.batch_spec.size * world_size
        self.weak_itr_batch_size = self.weak_sampler.batch_spec.size * world_size

        n_itr = self.get_n_itr()
        self.agent.to_device(self.affinity.get("cuda_idx", None))
        self.weak_agent.to_device(self.affinity.get("cuda_idx", None))

        if world_size > 1:
            self.agent.data_parallel()
            self.weak_agent.data_parallel()

        self.algo.initialize(
            agent=self.agent,
            n_itr=n_itr,
            batch_spec=self.sampler.batch_spec,
            mid_batch_reset=self.sampler.mid_batch_reset,
            examples=examples,
            world_size=world_size,
            rank=rank,
        )
        self.weak_algo.initialize(
            agent=self.weak_agent,
            n_itr=n_itr,
            batch_spec=self.weak_sampler.batch_spec,
            mid_batch_reset=self.weak_sampler.mid_batch_reset,
            examples=weak_examples,
            world_size=world_size,
            rank=rank,
        )
        self.initialize_logging()
        return n_itr

    def train_step(self, agent, sampler, algo, itr):
        agent.sample_mode(itr)
        samples, traj_infos = sampler.obtain_samples(itr)

        agent.train_mode(itr)
        opt_info = algo.optimize_agent(itr, samples)
        self.store_diagnostics(itr, traj_infos, opt_info)
        if (itr + 1) % self.log_interval_itrs == 0:
            eval_traj_infos, eval_time = self.evaluate_agent(itr, agent=agent, sampler=sampler)
            self.log_diagnostics(itr, eval_traj_infos, eval_time)

    def train(self):
        """
        Performs startup, evaluates the initial agent, then loops by
        alternating between ``sampler.obtain_samples()`` and
        ``algo.optimize_agent()``.  Pauses to evaluate the agent at the
        specified log interval.
        """
        n_itr = self.startup()
        with logger.prefix(f"itr #0 "):
            eval_traj_infos, eval_time = self.evaluate_agent(0)
            self.log_diagnostics(0, eval_traj_infos, eval_time)

            with logger.tabular_prefix("Weak"):
                eval_traj_infos, eval_time = self.evaluate_agent(0, agent=self.weak_agent, sampler=self.weak_sampler)
                self.log_diagnostics(0, eval_traj_infos, eval_time)

        for itr in range(n_itr):
            logger.set_iteration(itr)
            with logger.prefix(f"itr #{itr} "):
                self.train_step(agent=self.agent, sampler=self.sampler, algo=self.algo, itr=itr)
                if itr > 50:  # FIXME, exists because low >= hi error in replay buffer, also diff arbitrary
                    with logger.tabular_prefix("Weak"):
                        for _ in range(2):  # FIXME(arbitrary)
                            self.train_step(agent=self.weak_agent, sampler=self.weak_sampler, algo=self.weak_algo, itr=itr)

            if itr > 20:  # FIXME, exists because low >= hi error in replay buffer
                # for _ in range(self.sampler.batch_spec.B):
                for _ in range(2):  # FIXME(arbitrary)
                    samples_from_replay = self.algo.replay_buffer.sample_batch(self.algo.batch_size)
                    self.env_model.refit(samples_from_replay)

        self.shutdown()

    def evaluate_agent(self, itr, agent=None, sampler=None):
        """
        Record offline evaluation of agent performance, by ``sampler.evaluate_agent()``.
        """
        if not agent: agent = self.agent
        if not sampler: sampler = self.sampler

        # Almost unmodified code from base class
        if itr > 0:
            self.pbar.stop()

        if itr >= self.min_itr_learn - 1 or itr == 0:
            logger.log("Evaluating agent...")
            agent.eval_mode(itr)  # Might be agent in sampler.
            eval_time = -time.time()
            traj_infos = sampler.evaluate_agent(itr)
            eval_time += time.time()
        else:
            traj_infos = []
            eval_time = 0.0
        logger.log("Evaluation runs-1 complete.")
        return traj_infos, eval_time
