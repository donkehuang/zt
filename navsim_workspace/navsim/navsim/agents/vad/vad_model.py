from typing import Dict
import numpy as np
import torch
import torch.nn as nn

from navsim.agents.vad.vad_config import VADConfig
from navsim.agents.vad.vad_detectors import VAD
from navsim.common.enums import StateSE2Index
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex
# from mmdet.models


class VADModel(nn.Module):
    def __init__(self, config: VADConfig):

        super().__init__()

        self._config = config
        self._detector = VAD(config.model['use_grid_mask'],
                             config.model['img_backbone'],
                             config.model['img_neck'],
                             config.model['pts_bbox_head'])
        
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:


        # TODO donke
        print('-----------------')
        print("forward start")
        print('-----------------')
        # self._detector.vad_detector_forward()
    
        VAD(self._config.model['use_grid_mask'],
                            self._config.model['img_backbone'],
                            self._config.model['img_neck'],
                            self._config.model['pts_bbox_head'])

        print('-----------------')
        print("forward successfully")
        print('-----------------')

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
