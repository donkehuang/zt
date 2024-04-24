from typing import Any, List, Dict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataclasses import Scene


import torch


class VADFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self):
        pass

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        ego_status = agent_input.ego_statuses[-1]
        velocity = torch.tensor(ego_status.ego_velocity)
        acceleration = torch.tensor(ego_status.ego_acceleration)
        driving_command = torch.tensor(ego_status.driving_command)
        ego_state_feature = torch.cat([velocity, acceleration, driving_command], dim=-1)

        return {"ego_state": ego_state_feature}


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling):
        self._trajectory_sampling = trajectory_sampling

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self._trajectory_sampling.num_poses
        )
        return {"trajectory": torch.tensor(future_trajectory.poses)}


class VADAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        hidden_layer_dim: int,
        lr: float,
        checkpoint_path: str = None,
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self._checkpoint_path = checkpoint_path

        self._lr = lr

        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(8, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, self._trajectory_sampling.num_poses * 3),
        )
        self._model = None

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        self._model = torch.load(self._checkpoint_path)

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        # TODO: switch to all sensors
        # return SensorConfig.build_all_sensors()
        return SensorConfig.build_no_sensors()

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [
            TrajectoryTargetBuilder(
                trajectory_sampling=self._trajectory_sampling
            ),
        ]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [VADFeatureBuilder()]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        poses: torch.Tensor = self._mlp(features["ego_state"])
        return {"trajectory": poses.reshape(-1, self._trajectory_sampling.num_poses, 3)}

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])

    def get_optimizers(self): #  -> Optimizer | Dict[str, Optimizer | LRScheduler]:
        return torch.optim.Adam(self._mlp.parameters(), lr=self._lr)