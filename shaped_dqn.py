
import torch
from collections import namedtuple

from rlpyt.algos.base import RlAlgorithm
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.replays.non_sequence.frame import (UniformReplayFrameBuffer,
    PrioritizedReplayFrameBuffer, AsyncUniformReplayFrameBuffer,
    AsyncPrioritizedReplayFrameBuffer)
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import select_at_indexes, valid_mean
from rlpyt.algos.utils import valid_from_done

OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr"])
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done"])


class ShapedDQN(DQN):

    def __init__(self,
                 shaping_function,
                 discount=0.99, batch_size=32, min_steps_learn=int(5e4), delta_clip=1., replay_size=int(1e6),
                 replay_ratio=8, target_update_tau=1, target_update_interval=312, n_step_return=1, learning_rate=2.5e-4,
                 OptimCls=torch.optim.Adam, optim_kwargs=None, initial_optim_state_dict=None, clip_grad_norm=10.,
                 eps_steps=int(1e6), double_dqn=False, prioritized_replay=False, pri_alpha=0.6, pri_beta_init=0.4,
                 pri_beta_final=1., pri_beta_steps=int(50e6), default_priority=None, ReplayBufferCls=None,
                 updates_per_sync=1):
        self.shaping_function = shaping_function
        super().__init__(discount, batch_size, min_steps_learn, delta_clip, replay_size, replay_ratio,
                         target_update_tau, target_update_interval, n_step_return, learning_rate, OptimCls,
                         optim_kwargs, initial_optim_state_dict, clip_grad_norm, eps_steps, double_dqn,
                         prioritized_replay, pri_alpha, pri_beta_init, pri_beta_final, pri_beta_steps, default_priority,
                         ReplayBufferCls, updates_per_sync)

    def loss(self, samples):
        """
        Computes the Q-learning loss, based on: 0.5 * (Q - target_Q) ^ 2.
        Implements regular DQN or Double-DQN for computing target_Q values
        using the agent's target network.  Computes the Huber loss using
        ``delta_clip``, or if ``None``, uses MSE.  When using prioritized
        replay, multiplies losses by importance sample weights.

        Input ``samples`` have leading batch dimension [B,..] (but not time).

        Calls the agent to compute forward pass on training inputs, and calls
        ``agent.target()`` to compute target values.

        Returns loss and TD-absolute-errors for use in prioritization.

        Warning:
            If not using mid_batch_reset, the sampler will only reset environments
            between iterations, so some samples in the replay buffer will be
            invalid.  This case is not supported here currently.
        """
        qs = self.agent(*samples.agent_inputs)
        q = select_at_indexes(samples.action, qs)
        with torch.no_grad():
            target_qs = self.agent.target(*samples.target_inputs)
            if self.double_dqn:
                next_qs = self.agent(*samples.target_inputs)
                next_a = torch.argmax(next_qs, dim=-1)
                target_q = select_at_indexes(next_a, target_qs)
            else:
                target_q = torch.max(target_qs, dim=-1).values
        disc_target_q = (self.discount ** self.n_step_return) * target_q
        y = samples.return_ + self.shaping_function(samples) + (1 - samples.done_n.float()) * disc_target_q
        delta = y - q
        losses = 0.5 * delta ** 2
        abs_delta = abs(delta)
        if self.delta_clip is not None:  # Huber loss.
            b = self.delta_clip * (abs_delta - self.delta_clip / 2)
            losses = torch.where(abs_delta <= self.delta_clip, losses, b)
        if self.prioritized_replay:
            losses *= samples.is_weights
        td_abs_errors = abs_delta.detach()
        if self.delta_clip is not None:
            td_abs_errors = torch.clamp(td_abs_errors, 0, self.delta_clip)
        if not self.mid_batch_reset:
            # FIXME: I think this is wrong, because the first "done" sample
            # is valid, but here there is no [T] dim, so there's no way to
            # know if a "done" sample is the first "done" in the sequence.
            raise NotImplementedError
            # valid = valid_from_done(samples.done)
            # loss = valid_mean(losses, valid)
            # td_abs_errors *= valid
        else:
            loss = torch.mean(losses)

        return loss, td_abs_errors
