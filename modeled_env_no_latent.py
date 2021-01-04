from collections import namedtuple

from rlpyt.utils.logging import logger

import torch
import torch.optim
from rlpyt.envs.base import Env
from rlpyt.samplers.collections import Samples
from torch import nn

EnvStep = namedtuple("EnvStep",
                     ["observation", "reward", "done", "env_info"])
EnvInfo = namedtuple("EnvInfo", [])  # Define in env file.
EnvSpaces = namedtuple("EnvSpaces", ["observation", "action"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")


class ModeledEnv(Env):
    """
    The learning task, e.g. an MDP containing a transition function T(state,
    action)-->state'.  Has a defined observation space and action space.
    """
    def __init__(self, latent_step_model,
                 reward_model, termination_model, env_cls, steps_limit=200):
        # TODO check whether weight decay would be benefitial
        self.latent_step_model: nn.Module = latent_step_model.to(device)
        self.latent_step_model_optimizer = torch.optim.Adam(self.latent_step_model.parameters())

        self.reward_model: nn.Module = reward_model.to(device)
        self.reward_model_optimizer = torch.optim.Adam(self.reward_model.parameters())

        self.termination_model: nn.Module = termination_model.to(device)
        self.termination_model_optimizer = torch.optim.Adam(self.termination_model.parameters())

        self.state = None
        self.env: Env = env_cls()
        self.env.reset()

        self.cum_total_steps = 0
        self.total_steps = 0
        self.steps_limit = steps_limit
        self.total_refits = 0

        self.datapoints_total = 0
        self.datapoints_done = 0

        super().__init__()

    def refit(self, samples: Samples):
        self.datapoints_total += len(samples.action)
        self.datapoints_done += (samples.done_n == True).sum()

        self.latent_step_model_optimizer.zero_grad()
        self.reward_model_optimizer.zero_grad()
        self.termination_model_optimizer.zero_grad()

        obs = samples.agent_inputs.observation.to(device)
        obsprim = samples.target_inputs.observation.to(device)

        action = torch.unsqueeze(samples.action, dim=1).float().to(device)
        obs_action = torch.cat([obs, action], dim=1)
        obs_obsprim_action = torch.cat([obs, obsprim, action], dim=1)

        state_reconstruction_loss = torch.nn.SmoothL1Loss()\
            (obs + self.latent_step_model(obs_action), obsprim)
        reward_reconstruction_loss = torch.nn.SmoothL1Loss()\
            (self.reward_model(obs_obsprim_action), torch.unsqueeze(samples.return_, dim=1).to(device))
        done_weight = (self.datapoints_total / self.datapoints_done) if self.datapoints_done > 0 else 1.
        termination_reconstruction_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([done_weight]).float())\
            (self.termination_model(obs_obsprim_action), torch.unsqueeze(samples.done_n == True, dim=1).float().to(device))

        (state_reconstruction_loss + reward_reconstruction_loss + termination_reconstruction_loss).backward()

        if self.total_refits % 1e2 == 0:  # FIXME(1e2)
            logger._tf_summary_writer.add_scalar('Model/refit/state_reconstruction_loss', state_reconstruction_loss.item(), global_step=self.total_refits)
            logger._tf_summary_writer.add_scalar('Model/refit/reward_reconstruction_loss', reward_reconstruction_loss.item(), global_step=self.total_refits)
            logger._tf_summary_writer.add_scalar('Model/refit/termination_reconstruction_loss', termination_reconstruction_loss.item(), global_step=self.total_refits)
            logger._tf_summary_writer.add_scalar('Model/refit/reward', torch.mean(samples.return_).item(), global_step=self.total_refits)
        self.total_refits += 1

        self.latent_step_model_optimizer.step()
        self.reward_model_optimizer.step()
        self.termination_model_optimizer.step()

    def step(self, action):
        """
        Run on timestep of the environment's dynamics using the input action,
        advancing the internal state; T(state,action)-->state'.

        Args:
            action: An element of this environment's action space.

        Returns:
            observation: An element of this environment's observation space corresponding to the next state.
            reward (float): A scalar reward resulting from the state transition.
            done (bool): Indicates whether the episode has ended.
            info (namedtuple): Additional custom information.
        """
        self.total_steps += 1
        self.cum_total_steps += 1
        with torch.no_grad():
            action = torch.tensor([float(action)], dtype=torch.float32).to(device)
            s = self.state.clone().detach()
            self.state = s + self.latent_step_model(torch.cat([s, action]))
            sprim = self.state.clone().detach()

            obs_obsprim_action = torch.cat([s, sprim, action])
            reward = self.reward_model(obs_obsprim_action)
            termination_score = torch.sigmoid(self.termination_model(obs_obsprim_action))
            done = bool(termination_score >= 0.5) or self.total_steps % self.steps_limit == 0

            if self.cum_total_steps % 1e1 == 0:  # FIXME(1e1)
                logger._tf_summary_writer.add_scalar('Model/step/reward', reward.item(), global_step=self.cum_total_steps)
                logger._tf_summary_writer.add_scalar('Model/step/done', termination_score.item(), global_step=self.cum_total_steps)
                logger._tf_summary_writer.add_scalar('Model/traj_length', self.total_steps, global_step=self.cum_total_steps)

            if done:
                self.reset()

            return EnvStep(sprim, reward.item(), done, None)

    def reset(self):
        self.total_steps = 0
        with torch.no_grad():
            obs = self.env.reset()
            self.state = torch.tensor(obs).to(device)
            return self.state.to(cpu)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def spaces(self):
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )

    def close(self):
        self.env.close()
