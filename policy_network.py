"""
    Traj generation policy (flow matching version) from NavDP
    https://github.com/InternRobotics/NavDP
    https://github.com/haidog-yaqub/MeanFlow

    by houjunyi, 2025-09-24
"""
import torch
import torch.nn as nn
import numpy as np
from models.policy_backbone import *
from meanflow import MeanFlow
from models.dit import MFDiT

class NavDP_Policy_MF(nn.Module):
    def __init__(self,
                 image_size=224,
                 memory_size=8,
                 predict_size=24,
                 token_dim=384,
                 channels=3,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.predict_size = predict_size
        self.token_dim = token_dim

        # 编码器
        self.rgbd_encoder = NavDP_RGBD_Backbone(image_size, token_dim, memory_size=memory_size, device=device)
        self.point_encoder = nn.Linear(3, token_dim)
        self.pixel_encoder = NavDP_PixelGoal_Backbone(image_size, token_dim, device=device)
        self.image_encoder = NavDP_ImageGoal_Backbone(image_size, token_dim, device=device)

        # MFDiT + MeanFlow
        self.mfdit = MFDiT(
            input_size=32,
            patch_size=2,
            in_channels=3,   # 轨迹作为 "图像"
            dim=token_dim,
            depth=8,
            num_heads=8,
            num_classes=None,
        ).to(device)

        self.meanflow = MeanFlow(
            channels=3,
            image_size=32,
            num_classes=None,
            flow_ratio=0.5,
            time_dist=['lognorm', -0.4, 1.0],
            cfg_ratio=0.0,
            cfg_scale=1.0,
            cfg_uncond=None
        )

    def predict_pointgoal_action(self, goal_point, input_images, input_depths, sample_num=16):
        """
        使用 MeanFlow + MFDiT 生成 point-goal 动作
        """
        with torch.no_grad():
            # 编码条件
            tensor_point_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self.device)
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)

            cond_embed = torch.cat([pointgoal_embed, rgbd_embed], dim=1)  # (B, L, D)

            # 调用 meanflow.sample
            model_module = self.mfdit
            # 这里生成“轨迹图像”，你可以把轨迹 reshape 成 (B, 3, H, W)，再交给 meanflow
            traj_samples = self.meanflow.sample(model_module, cond=cond_embed, batch_size=sample_num)

            # 转回轨迹 (B, sample_num, T, 3)
            B = goal_point.shape[0]
            all_trajectory = traj_samples.reshape(B, sample_num, self.predict_size, 3)

            # critic：末端距离
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            critic_values = trajectory_length.clone()

            # 选正负轨迹
            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(B).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]

            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            negative_trajectory = all_trajectory[batch_indices, topk_indices]

            return (all_trajectory.cpu().numpy(),
                    critic_values.cpu().numpy(),
                    positive_trajectory.cpu().numpy(),
                    negative_trajectory.cpu().numpy())
