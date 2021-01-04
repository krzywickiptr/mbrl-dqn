from functools import reduce

import torch
import torch.nn
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.agents.pg.categorical import CategoricalPgAgent

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel

class MlpMixin:
    def make_env_to_model_kwargs(self, env_spaces):
        return dict(input_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)


class MlpDqnModel(torch.nn.Module):
    """feeding an MLP for Q-value outputs for the action set."""

    def __init__(
            self,
            output_size,
            input_size=None,
            input_shape=None,
            fc_sizes=512,
            dueling=False,
            **kwargs,
            ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        assert input_shape or input_size
        self.dueling = dueling
        if input_shape:
            input_size = reduce(lambda x, y: x * y, input_shape)

        if dueling:
            self.head = DuelingHeadModel(input_size, fc_sizes, output_size)
        else:
            self.head = MlpModel(input_size, fc_sizes, output_size)

    def forward(self, observation, prev_action=None, prev_reward=None):
        """
        Compute action Q-value estimates from input state.
        Infers leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Convolution layers process as [T*B,
        image_shape[0], image_shape[1],...,image_shape[-1]], with T=1,B=1 when not given.  Expects uint8 images in
        [0,255] and converts them to float32
        Used in both sampler and in algorithm (both via the agent).
        """
        obs = observation.type(torch.float)  # Expect torch.uint8 inputs
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        if len(observation.shape) < 3:
            reshape = (observation.shape[0], -1) if len(observation.shape) == 2 else (-1,)
            view = obs.view(*reshape)
            return self.head(view)

        lead_dim, T, B, _ = infer_leading_dims(obs, 3)

        q = self.head(obs.view(T * B, -1))

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)
        return q


class MlpDqnAgent(MlpMixin, DqnAgent):
    def __init__(self, ModelCls=MlpDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

class MlpCategoricalPgAgent(MlpMixin, CategoricalPgAgent):
    def __init__(self, ModelCls=MlpDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)