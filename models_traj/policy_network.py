import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from navdp_test.policy_backbone import *
from scipy.interpolate import interp1d

class NavDP_Policy_Flow(nn.Module):
    def __init__(self,
                 image_size=224,
                 memory_size=8,
                 predict_size=24,
                 temporal_depth=8,
                 heads=8,
                 token_dim=384,
                 channels=3,
                 device='cuda:0',
                 solver_steps=50):   # 新增：积分步数（采样时用）
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.temporal_depth = temporal_depth
        self.attention_heads = heads
        self.input_channels = channels
        self.token_dim = token_dim
        self.solver_steps = solver_steps  # 用于采样积分的步数
        
        # input encoders (保持不变)
        self.rgbd_encoder = NavDP_RGBD_Backbone(image_size, token_dim, memory_size=memory_size, device=device)
        self.point_encoder = nn.Linear(3, self.token_dim)
        self.pixel_encoder = NavDP_PixelGoal_Backbone(image_size, token_dim, device=device)
        self.image_encoder = NavDP_ImageGoal_Backbone(image_size, token_dim, device=device)
        
        # fusion layers (保持不变)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = token_dim,
                                                        nhead = heads,
                                                        dim_feedforward = 4 * token_dim,
                                                        activation = 'gelu',
                                                        batch_first = True,
                                                        norm_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer = self.decoder_layer,
                                             num_layers = self.temporal_depth)
        
        # input embed (actions -> tokens)
        self.input_embed = nn.Linear(3, token_dim) # encode the actions for velocity/critic
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, memory_size * 16 + 4) # time,point,image,pixel,input
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size) 
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.layernorm = nn.LayerNorm(token_dim)
        
        # heads
        # 注意： velocity head 输出的是 与动作相同维度的速度（predict_size x 3）
        self.velocity_head = nn.Linear(token_dim, 3)
        self.critic_head = nn.Linear(token_dim, 1)
        
        # 旧的 DDPM scheduler 被移除：flow matching 不需要 DDPMScheduler
        # 但保留一些原先的 mask 构造逻辑（如果 decoder 中需要）
        self.tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = self.tgt_mask.float().masked_fill(self.tgt_mask == 0, float('-inf')).masked_fill(self.tgt_mask == 1, float(0.0))
        self.cond_critic_mask = torch.zeros((predict_size,4 + memory_size * 16))
        self.cond_critic_mask[:,0:4] = float('-inf')
    
    # ---------- 核心改变：预测速度场（而不是噪声） ----------
    def predict_velocity(self, actions, t, goal_embed, rgbd_embed):
        """
        actions: (B, T, 3) - 当前点（状态） —— 在采样时这是当前 x_t
        t: tensor shape (B,) or scalar time in [0,1] (float tensor)
        goal_embed: (B, M, token_dim) or shape consistent with original cond embedding (we expect 1 token per goal here)
        rgbd_embed: (B, N, token_dim)
        返回: velocity same shape as actions: (B, T, 3)
        """
        # embed actions
        action_embeds = self.input_embed(actions)  # (B, T, token_dim)
        # time embedding (per-batch)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32, device=self.device)
        if t.ndim == 1 and t.shape[0] == actions.shape[0]:
            time_embeds = self.time_emb(t.to(self.device)).unsqueeze(1)  # (B,1,token_dim)
        else:
            # allow scalar passed in
            time_embeds = self.time_emb(t.to(self.device)).unsqueeze(1).repeat(actions.shape[0], 1, 1)
        
        # build cond embedding: consistent with original code ordering:
        cond_embedding = torch.cat([time_embeds, goal_embed, goal_embed, goal_embed, rgbd_embed], dim=1)
        cond_embedding = cond_embedding + self.cond_pos_embed(cond_embedding)
        
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        # pass through transformer decoder: tgt = action tokens, memory = cond
        output = self.decoder(tgt = input_embedding, memory = cond_embedding, tgt_mask = self.tgt_mask.to(self.device))
        output = self.layernorm(output)
        velocity = self.velocity_head(output)  # (B, T, 3)
        return velocity

    def predict_mix_velocity(self, actions, t, goal_embeds, rgbd_embed):
        """
        多条件混合版本，goal_embeds 是 list/tuple [image_goal, point_goal, image_goal]（与原代码相仿）
        """
        action_embeds = self.input_embed(actions)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32, device=self.device)
        time_embeds = self.time_emb(t.to(self.device)).unsqueeze(1).tile((actions.shape[0],1,1))
        cond_embedding = torch.cat([time_embeds, goal_embeds[0], goal_embeds[1], goal_embeds[2], rgbd_embed], dim=1)
        cond_embedding = cond_embedding + self.cond_pos_embed(cond_embedding)
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(tgt = input_embedding, memory = cond_embedding, tgt_mask = self.tgt_mask.to(self.device))
        output = self.layernorm(output)
        velocity = self.velocity_head(output)
        return velocity

    # ---------- critic 保持几乎不变 ----------
    def predict_critic(self, predict_trajectory, rgbd_embed):
        nogoal_embed = torch.zeros_like(rgbd_embed[:,0:1])
        action_embeddings = self.input_embed(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        cond_embeddings = torch.cat([nogoal_embed,nogoal_embed,nogoal_embed,nogoal_embed,rgbd_embed],dim=1) +  self.cond_pos_embed(torch.cat([nogoal_embed,nogoal_embed,nogoal_embed,nogoal_embed,rgbd_embed],dim=1))
        critic_output = self.decoder(tgt = action_embeddings, memory = cond_embeddings, memory_mask = self.cond_critic_mask.to(self.device))
        critic_output = self.layernorm(critic_output)
        critic_output = self.critic_head(critic_output.mean(dim=1))[:,0]
        return critic_output

    # ---------- 采样 helper：用显式 Euler 积分 velocity field（1.0 -> 0.0） ----------
    def integrate_velocity_field(self, x_init, cond_goal_embed, rgbd_embed, mix_goals=None):
        """
        x_init: (B, T, 3) at time t=1.0
        cond_goal_embed: goal embedding aligned for predict_velocity (shape (B, M, token_dim))
        rgbd_embed: (B, N, token_dim)
        mix_goals: optional list for predict_mix_velocity
        返回: x at t=0 (动作序列)
        """
        B = x_init.shape[0]
        x = x_init.clone()
        # make time schedule (from 1.0 down to 0.0)
        ts = torch.linspace(1.0, 0.0, steps=self.solver_steps, device=self.device)
        # integrate (simple explicit Euler). 可替换为 RK4 提升精度
        for i in range(len(ts)-1):
            t = ts[i]
            dt = ts[i] - ts[i+1]
            if mix_goals is None:
                v = self.predict_velocity(x, torch.full((B,), t, device=self.device), cond_goal_embed, rgbd_embed)
            else:
                v = self.predict_mix_velocity(x, torch.full((B,), t, device=self.device), mix_goals, rgbd_embed)
            # Euler step: x_{t+dt} = x_t + v * dt
            x = x + v * dt
        return x

    # ---------- 将 predict_*_action 中的 DDPM loop 替换为 flow-matching 积分 ----------
    def predict_pointgoal_action(self, goal_point, input_images, input_depths, sample_num=16):
        with torch.no_grad():
            tensor_point_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self.device)
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)  # (B, Ncond, token_dim)
            pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)  # (B,1,token_dim)
    
            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            pointgoal_embed = torch.repeat_interleave(pointgoal_embed, sample_num, dim=0)
            
            # 初始噪声（t=1.0）
            naction = torch.randn((sample_num * goal_point.shape[0], self.predict_size, 3), device=self.device)
            # 积分 velocity field 得到轨迹（t->0）
            naction = self.integrate_velocity_field(naction, pointgoal_embed, rgbd_embed)
            
            critic_values = self.predict_critic(naction, rgbd_embed)
            critic_values = critic_values.reshape(goal_point.shape[0], sample_num)
            
            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(goal_point.shape[0], sample_num, self.predict_size, 3)
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            # 轨迹长度小于0.1则不学习
            all_trajectory[trajectory_length < 0.1] = all_trajectory[trajectory_length < 0.1] * torch.tensor([[[0,0,1.0]]], device=all_trajectory.device)
            
            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_point.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]
            
            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_point.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]
            
            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()
    
    def predict_imagegoal_action(self, goal_image, input_images, input_depths, sample_num=16):
        with torch.no_grad():
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            imagegoal_embed = self.image_encoder(np.concatenate((goal_image, input_images[:,-1]), axis=-1)).unsqueeze(1)
    
            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            imagegoal_embed = torch.repeat_interleave(imagegoal_embed, sample_num, dim=0)
            
            naction = torch.randn((sample_num * goal_image.shape[0], self.predict_size, 3), device=self.device)
            naction = self.integrate_velocity_field(naction, imagegoal_embed, rgbd_embed)
            
            critic_values = self.predict_critic(naction, rgbd_embed)
            critic_values = critic_values.reshape(goal_image.shape[0], sample_num)
            
            all_trajectory = torch.cumsum(naction / 40.0, dim=1)
            all_trajectory = all_trajectory.reshape(goal_image.shape[0], sample_num, self.predict_size, 3)
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * torch.tensor([[[0,0,1.0]]], device=all_trajectory.device)
            
            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]
            
            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]
            
            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()
    
    def predict_pixelgoal_action(self, goal_image, input_images, input_depths, sample_num=16):
        with torch.no_grad():
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            pixelgoal_embed = self.pixel_encoder(np.concatenate((goal_image[:,:,:,None], input_images[:,-1]), axis=-1)).unsqueeze(1)
    
            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            pixelgoal_embed = torch.repeat_interleave(pixelgoal_embed, sample_num, dim=0)
            
            naction = torch.randn((sample_num * goal_image.shape[0], self.predict_size, 3), device=self.device)
            naction = self.integrate_velocity_field(naction, pixelgoal_embed, rgbd_embed)
            
            critic_values = self.predict_critic(naction, rgbd_embed)
            critic_values = critic_values.reshape(goal_image.shape[0], sample_num)
            
            all_trajectory = torch.cumsum(naction / 40.0, dim=1)
            all_trajectory = all_trajectory.reshape(goal_image.shape[0], sample_num, self.predict_size, 3)
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * torch.tensor([[[0,0,1.0]]], device=all_trajectory.device)
            
            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]
            
            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]
            
            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()

    def predict_nogoal_action(self, input_images, input_depths, sample_num=16):
        with torch.no_grad():
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            nogoal_embed = torch.zeros_like(rgbd_embed[:,0:1])
            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            nogoal_embed = torch.repeat_interleave(nogoal_embed, sample_num, dim=0)
           
            naction = torch.randn((sample_num * input_images.shape[0], self.predict_size, 3), device=self.device)
            naction = self.integrate_velocity_field(naction, nogoal_embed, rgbd_embed)
            
            critic_values = self.predict_critic(naction, rgbd_embed)
            critic_values = critic_values.reshape(input_images.shape[0], sample_num)
            
            all_trajectory = torch.cumsum(naction / 40.0, dim=1)
            all_trajectory = all_trajectory.reshape(input_images.shape[0], sample_num, self.predict_size, 3)

            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            # 我们对非常短的轨迹打低分（同原逻辑）
            critic_values[torch.where(trajectory_length < 1.0)] -= 10.0
            
            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(input_images.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]
            
            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(input_images.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]
            
            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()
        
    def predict_ip_action(self, goal_point, goal_image, input_images, input_depths, sample_num=16):
        with torch.no_grad():
            tensor_point_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self.device)
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            imagegoal_embed = self.image_encoder(np.concatenate((goal_image, input_images[:,-1]), axis=-1)).unsqueeze(1)
            pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)
            
            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            pointgoal_embed = torch.repeat_interleave(pointgoal_embed, sample_num, dim=0)
            imagegoal_embed = torch.repeat_interleave(imagegoal_embed, sample_num, dim=0)
            
            naction = torch.randn((sample_num * goal_image.shape[0], self.predict_size, 3), device=self.device)
            naction = self.integrate_velocity_field(naction, imagegoal_embed, rgbd_embed, mix_goals=[imagegoal_embed, pointgoal_embed, imagegoal_embed])
            
            critic_values = self.predict_critic(naction, rgbd_embed)
            critic_values = critic_values.reshape(goal_image.shape[0], sample_num)
            
            all_trajectory = torch.cumsum(naction / 40.0, dim=1)
            all_trajectory = all_trajectory.reshape(goal_image.shape[0], sample_num, self.predict_size, 3)
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * torch.tensor([[[0,0,1.0]]], device=all_trajectory.device)
            
            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]
            
            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(goal_image.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]
            
            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy()

    # ---------- 训练时用的基础 flow-matching loss 实现（可替换/扩展） ----------
    def compute_flow_matching_loss(self, a_start, a_target, goal, goal_embed, rgbd_embed, mix_goals=None):
        B, T, _ = a_start.shape
        t = torch.rand(B, device=self.device)  # in (0,1)
        t_ = t.view(B,1,1)

        a_target_interp = self.process_target_trajectory(a_target, a_start)
        
        # t=0是干净数据，t=1是纯高斯噪声数据
        x_t = (1.0 - t_) * a_target_interp + t_ * a_start # (B,T,3)

        # predict velocity
        if mix_goals is None:
            v_pred = self.predict_velocity(x_t, t, goal_embed, rgbd_embed)  # (B,T,3)
        else:
            v_pred = self.predict_mix_velocity(x_t, t, mix_goals, rgbd_embed)

        # target velocity
        v_target = a_target_interp - a_start

        # mean squared error
        loss = F.mse_loss(v_pred, v_target)
        return loss

    def process_target_trajectory(self, a_target, a_start):
        B, T, point_dim = a_target.shape
        B_start, T_start, _ = a_start.shape
        # =============================
        # 1. 提取有效部分 (非零动作)
        # =============================
        mask_valid = (a_target.abs().sum(dim=-1) != 0)  # (B, T), True 表示有效

        a_target_valid = []
        for b in range(B):
            valid_idx = mask_valid[b].nonzero(as_tuple=False).squeeze(-1)
            a_target_b = a_target[b, valid_idx]  # (L_b, 3)
            a_target_valid.append(a_target_b)
        
        # =============================
        # 2. 将动作序列转换为轨迹点坐标
        # =============================
        a_target_interp = torch.zeros_like(a_target)  # (B, T, 3)
        for b in range(B):
            valid_len = a_target_valid[b].shape[0]
            if valid_len == 0:
                # 原地停止时，直接填充整个序列
                a_target_interp[b] = a_target[b]
            elif valid_len == 1:
                # 只有一个有效动作时，除以动作步再填充整个序列
                a_target_interp[b] = (a_target_valid[b] / T).expand(T, 3)
            else:
                # 将动作序列转换为轨迹点坐标（累积和）
                traj_points = torch.cumsum(a_target_valid[b], dim=0)  # (L_b, 3)
                
                # 添加起点 (0, 0, 0)
                traj_points_with_start = torch.cat([
                    torch.zeros(1, 3, device=self.device), 
                    traj_points
                ], dim=0)  # (L_b+1, 3)
                
                # =============================
                # 3. 在轨迹点坐标层面进行插值
                # =============================
                if valid_len == 2:
                    # 只有两个点时，使用线性插值
                    t_orig = torch.linspace(0, 1, steps=valid_len+1, device=self.device).cpu().numpy()
                    t_new = torch.linspace(0, 1, steps=T_start+1, device=self.device).cpu().numpy()
                    traj_interp = torch.zeros(T_start+1, 3, device=self.device)
                    for j in range(point_dim):
                        traj_interp[:, j] = torch.tensor(
                            np.interp(t_new, t_orig, traj_points_with_start[:, j].cpu().numpy())
                        ).to(self.device)
                else:
                    # 使用三次样条插值
                    t_orig = torch.linspace(0, 1, steps=valid_len+1, device=self.device).cpu().numpy()
                    t_new = torch.linspace(0, 1, steps=T_start+1, device=self.device).cpu().numpy()
                    traj_interp = torch.zeros(T_start+1, 3, device=self.device)
                    for j in range(point_dim):
                        f = interp1d(t_orig, traj_points_with_start[:, j].cpu().numpy(), kind='cubic')
                        traj_interp[:, j] = torch.tensor(f(t_new), dtype=torch.float32, device=self.device)
                
                # =============================
                # 4. 插值后的轨迹点直接作为动作序列
                # =============================
                a_target_interp[b] = traj_interp[1:] - traj_interp[:-1]  # (T_start, 3)
        return a_target_interp
