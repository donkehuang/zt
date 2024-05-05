from typing import Dict
import numpy as np
import torch
import torch.nn as nn

from navsim.agents.vad.vad_config import VADConfig
from navsim.agents.vad.vad_detectors import VAD
from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from navsim.common.enums import StateSE2Index
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex
# from mmdet.models


class VADModel(nn.Module):
    def __init__(self, config: VADConfig):

        super().__init__()

        self._config = config
        print(config.model['img_backbone'])
        # self._detector = VAD(config.model)
        self._detector = VAD(config.model['use_grid_mask'],
                             config.model['img_backbone'],
                             config.model['img_neck'],
                             config.model['pts_bbox_head'])
        
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:


        # TODO donke
        print("model build by vad")
        # camera_feature: torch.Tensor = features["camera_feature"]
        # lidar_feature: torch.Tensor = features["lidar_feature"]
        # status_feature: torch.Tensor = features["status_feature"]

        # batch_size = status_feature.shape[0]

        # bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)

        # bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        # bev_feature = bev_feature.permute(0, 2, 1)
        # status_encoding = self._status_encoding(status_feature)

        # keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        # keyval += self._keyval_embedding.weight[None, ...]

        # query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        # query_out = self._tf_decoder(query, keyval)

        # bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        # trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        # output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        # trajectory = self._trajectory_head(trajectory_query)
        # output.update(trajectory)

        # agents = self._agent_head(agents_query)
        # output.update(agents)

        return 


class AgentHead(nn.Module):
    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = (
            agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        )
        agent_states[..., BoundingBox2DIndex.HEADING] = (
            agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi
        )

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}


class TrajectoryHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, num_poses * StateSE2Index.size()),
        )

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        return {"trajectory": poses}
