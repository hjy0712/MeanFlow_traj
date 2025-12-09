import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from scipy.interpolate import interp1d
from navdp_test.policy_backbone import *


# ---------- 工具函数 ----------
def stopgrad(x):
    return x.detach()

def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    delta_sq = torch.mean(error ** 2, dim=(1, 2), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()


# ---------- MeanFlow 模型 ----------
class NavDP_Policy_MeanFlow(nn.Module):
    def __init__(self,
                 image_size=224,
                 memory_size=8,
                 predict_size=24,
                 temporal_depth=8,
                 heads=8,
                 token_dim=384,
                 channels=3,
                 device='cuda:0',
                 solver_steps=1,
                 flow_ratio=0.2, # 0.2 表示 20% 的轨迹被选择为t和r相等
                 time_dist=('lognorm', -0.4, 1.0),
                 jvp_api='autograd'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.memory_size = memory_size
        self.predict_size = predict_size
        self.temporal_depth = temporal_depth
        self.attention_heads = heads
        self.input_channels = channels
        self.token_dim = token_dim
        self.solver_steps = solver_steps

        # flow matching hyperparams
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.jvp_api = jvp_api

        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        else:
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

        # ---------- Encoder / Decoder ----------
        self.rgbd_encoder = NavDP_RGBD_Backbone(image_size, token_dim, memory_size=memory_size, device=device)
        self.point_encoder = nn.Linear(3, self.token_dim)
        self.pixel_encoder = NavDP_PixelGoal_Backbone(image_size, token_dim, device=device)
        self.image_encoder = NavDP_ImageGoal_Backbone(image_size, token_dim, device=device)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim, nhead=heads, dim_feedforward=4 * token_dim,
            activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=temporal_depth)

        self.input_embed = nn.Linear(3, token_dim)
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, memory_size * 16 + 6)
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.layernorm = nn.LayerNorm(token_dim)

        self.velocity_head = nn.Linear(token_dim, 3)
        self.critic_head = nn.Linear(token_dim, 1)

        self.tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = self.tgt_mask.float().masked_fill(self.tgt_mask == 0, float('-inf')).masked_fill(self.tgt_mask == 1, float(0.0))
        self.cond_critic_mask = torch.zeros((predict_size, 4 + memory_size * 16))
        self.cond_critic_mask[:, 0:4] = float('-inf')


    # ---------- MeanFlow预测速度 ----------
    def predict_velocity(self, actions, t, r, goal_embed, rgbd_embed):
        """
        MeanFlow 版本：考虑时间区间 (r, t)
        actions: (B, T, 3) - 当前状态 z_t
        t, r: (B,) - 当前与参考时间点
        goal_embed: (B, M, D)
        rgbd_embed: (B, N, D)
        输出: (B, T, 3)
        """
        B = actions.shape[0]

        # --- embed actions ---
        action_embeds = self.input_embed(actions)  # (B, T, D)

        # --- embed times ---
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32, device=self.device)
        if not isinstance(r, torch.Tensor):
            r = torch.tensor([r], dtype=torch.float32, device=self.device)

        # time embeddings
        t_emb = self.time_emb(t.to(self.device)).unsqueeze(1)  # (B, 1, D)
        r_emb = self.time_emb(r.to(self.device)).unsqueeze(1)  # (B, 1, D)

        # time-diff embedding (关键！MeanFlow核心思想)
        delta_t = (t - r)  # (B,)
        delta_emb = self.time_emb(delta_t.to(self.device)).unsqueeze(1)  # (B, 1, D)

        # --- build conditional embedding ---
        # 保留 goal / rgbd 结构一致，加入时间区间信息
        cond_embedding = torch.cat([
            t_emb,        # 当前时刻
            r_emb,        # 起始时刻
            delta_emb,    # 时间间隔
            goal_embed, goal_embed, goal_embed, rgbd_embed
        ], dim=1)
        cond_embedding = cond_embedding + self.cond_pos_embed(cond_embedding)

        # --- transformer forward ---
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(
            tgt=input_embedding,
            memory=cond_embedding,
            tgt_mask=self.tgt_mask.to(self.device)
        )
        output = self.layernorm(output)
        velocity = self.velocity_head(output)  # (B, T, 3)
        return velocity



    # ---------- MeanFlow 核心 ----------
    def sample_t_r(self, batch_size, device):
        dist_type, mu, sigma = self.time_dist
        if dist_type == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        else:
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]
        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r


    def compute_meanflow_loss(self, model_fn, a_start, a_target, goal_embed, rgbd_embed):
        B, T, _ = a_start.shape
        device = a_start.device

        a_target_interp = self.process_target_trajectory(a_target, a_start) # (B,T,3)

        t, r = self.sample_t_r(B, device)
        t_ = t.view(B, 1, 1)  # 从 (B,1,1,1) 改为 (B,1,1)
        r_ = r.view(B, 1, 1)  # 从 (B,1,1,1) 改为 (B,1,1)

        v = a_start - a_target_interp
        z = (1 - t_) * a_target_interp + t_ * a_start   # t=0 -> clean, t=1 -> noise

        model_partial = partial(model_fn)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r, goal_embed, rgbd_embed),
            (z, t, r),
            (v, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v - (t_ - r_) * dudt
        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
        mse = (stopgrad(error) ** 2).mean()
        return loss, mse

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


    # ---------- 将 predict_*_action 中的 DDPM loop 替换为 flow-matching 积分 ----------
    def predict_pointgoal_action(self, goal_point, input_images, input_depths, sample_num=16):
        """
        MeanFlow版本的轨迹生成函数。
        从随机噪声出发（t=1.0），通过MeanFlow积分生成到t=0.0的轨迹。
        """
        with torch.no_grad():
            B = goal_point.shape[0]
            tensor_point_goal = torch.as_tensor(goal_point, dtype=torch.float32, device=self.device)
            rgbd_embed = self.rgbd_encoder(input_images, input_depths)
            pointgoal_embed = self.point_encoder(tensor_point_goal).unsqueeze(1)

            # 重复采样样本（相当于批量 * sample_num）
            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            pointgoal_embed = torch.repeat_interleave(pointgoal_embed, sample_num, dim=0)

            # 初始化随机动作 (t=1.0)
            naction = torch.randn((sample_num * B, self.predict_size, 3), device=self.device)

            # ===== MeanFlow 积分 =====
            ts = torch.linspace(1.0, 0.0, steps=self.solver_steps + 1, device=self.device)
            for i in range(len(ts) - 1):
                t = ts[i].expand(sample_num * B)      # 当前时间
                r = ts[i + 1].expand(sample_num * B)  # 下一步时间
                delta_t = (t - r).view(-1, 1, 1)     # 时间步长
                
                # MeanFlow更新： x_r = x_t - (t - r) * u_theta(x_t, t, r)
                v = self.predict_velocity(naction, t, r, pointgoal_embed, rgbd_embed)
                naction = naction - delta_t * v

            # ===== critic & ranking =====
            critic_values = self.predict_critic(naction, rgbd_embed)
            critic_values = critic_values.view(B, sample_num)

            # 累积积分成轨迹
            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.view(B, sample_num, self.predict_size, 3)

            # 去掉太短的轨迹
            trajectory_length = all_trajectory[:, :, -1, 0:2].norm(dim=-1)
            mask = trajectory_length < 0.1
            all_trajectory[mask] = all_trajectory[mask] * torch.tensor([[[0, 0, 1.0]]], device=self.device)

            # ===== top-k选轨迹 =====
            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:, :2]
            batch_indices = torch.arange(B).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]

            sorted_indices = critic_values.argsort(dim=1)
            topk_indices = sorted_indices[:, :2]
            batch_indices = torch.arange(B).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]

            return (
                all_trajectory.cpu().numpy(),
                critic_values.cpu().numpy(),
                positive_trajectory.cpu().numpy(),
                negative_trajectory.cpu().numpy(),
            )

    # # ---------- 训练时用的基础 flow-matching loss 实现 ----------
    # def compute_flow_matching_loss(self, a_start, a_target, goal, goal_embed, rgbd_embed, mix_goals=None):
    #     B, T, _ = a_start.shape
    #     t = torch.rand(B, device=self.device)  # in (0,1)
    #     t_ = t.view(B,1,1)

    #     a_target_interp = self.process_target_trajectory(a_target, a_start)
        
    #     # t=0是干净数据，t=1是纯高斯噪声数据
    #     x_t = (1.0 - t_) * a_target_interp + t_ * a_start # (B,T,3)

    #     # predict velocity
    #     if mix_goals is None:
    #         v_pred = self.predict_velocity(x_t, t, goal_embed, rgbd_embed)  # (B,T,3)
    #     else:
    #         v_pred = self.predict_mix_velocity(x_t, t, mix_goals, rgbd_embed)

    #     # target velocity
    #     v_target = a_target_interp - a_start

    #     # mean squared error
    #     loss = F.mse_loss(v_pred, v_target)
    #     return loss

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

