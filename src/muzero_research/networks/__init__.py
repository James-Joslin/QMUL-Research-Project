"""Network construction without hidden factory classes or runtime initialisation."""

from ..config import ExperimentConfig
from ..contracts import WorldModel
from .models import MuZeroFullyConnectedNetwork, MuZeroResidualNetwork


def build_network(config: ExperimentConfig) -> WorldModel:
    config.validate()
    e, n = config.environment, config.network
    common = (e.observation_shape, e.stacked_observations, len(e.action_space))
    if n.kind == "fullyconnected":
        return MuZeroFullyConnectedNetwork(
            *common,
            n.encoding_size,
            n.fc_reward_layers,
            n.fc_value_layers,
            n.fc_policy_layers,
            n.fc_representation_layers,
            n.fc_dynamics_layers,
            n.support_size,
        )
    return MuZeroResidualNetwork(
        *common,
        n.blocks,
        n.channels,
        n.reduced_channels_reward,
        n.reduced_channels_value,
        n.reduced_channels_policy,
        n.resnet_fc_reward_layers,
        n.resnet_fc_value_layers,
        n.resnet_fc_policy_layers,
        n.support_size,
        n.downsample,
    )
