
"""
轨迹可视化函数
"""

import numpy as np
import torch
import cv2
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import matplotlib.pyplot as plt

def log_plot_txt(image, depth, goal, traj, reward, reward_info):

        log_image = image[0, 0, 0]
        log_depth = depth[0, 0, 0]
        log_goal = goal[0, 0, 0]    # [x, y]
        log_traj = traj[0, :10]          # (traj_per_env, 24, 3)
        log_traj_reward = reward[0] # (traj_per_env, )
        log_traj_reward_info = reward_info[0] # (traj_per_env, 3)

        traj_xy = np.array(log_traj)
        xs, ys = traj_xy[:, :, 0], traj_xy[:, :, 1]

        # 固定 reward 范围 [-30, 30]
        vmin, vmax = log_traj_reward.min(), log_traj_reward.max()
        log_traj_reward_clipped = np.clip(log_traj_reward, vmin, vmax)

        # 根据 reward 设置颜色（高红低蓝）
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('coolwarm')
        colors = cmap(norm(log_traj_reward_clipped))

        # 创建空白图像 (比如 256x256)
        canvas_size = 256
        traj_img = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

        def world_to_img(x, y):
            # x: -1 → canvas_size-1 (底部), 6 → 0 (顶部)
            iy = int((1.0 - (x + 1) / 7.0) * (canvas_size - 1))
            iy = np.clip(iy, 0, canvas_size - 1)
            
            # y: -6 → canvas_size-1 (右边), 6 → 0 (左边)
            ix = int((1.0 - (y + 6) / 12.0) * (canvas_size - 1))
            ix = np.clip(ix, 0, canvas_size - 1)
    
            return ix, iy

        # 画自车 (0,0)
        car_x, car_y = world_to_img(0, 0)
        cv2.circle(traj_img, (car_x, car_y), 5, (0, 0, 255), -1)
        cv2.putText(traj_img, "(0,0)", (car_x+5, car_y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        # 画 goal
        goal_x, goal_y = world_to_img(log_goal[0], log_goal[1])
        cv2.circle(traj_img, (goal_x, goal_y), 5, (0,255,0), -1)
        cv2.putText(traj_img, f"({log_goal[0]:.1f},{log_goal[1]:.1f})", 
                    (goal_x+5, goal_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        # 画轨迹点
        for traj_idx, (x_seq, y_seq, c) in enumerate(zip(xs, ys, colors)):
            color_bgr = (int(c[2]*255), int(c[1]*255), int(c[0]*255))
            for px, py in zip(x_seq, y_seq):
                ix, iy = world_to_img(px, py)
                cv2.circle(traj_img, (ix, iy), 1, color_bgr, -1)

        # --------- 拼接并保存 ---------

        # 1. RGB 图像从 [0,1] 转换为 [0,255] 并转uint8
        rgb_img = (np.clip(log_image, 0.0, 1.0) * 255).astype(np.uint8)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        # 2. 深度图映射到[0,255]可视化
        depth_vis = (log_depth / (log_depth.max() + 1e-6) * 255).astype(np.uint8)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        # 3. 调整大小
        rgb_resized = cv2.resize(rgb_img, (canvas_size, canvas_size))
        depth_resized = cv2.resize(depth_vis, (canvas_size, canvas_size))
        traj_resized = cv2.resize(traj_img, (canvas_size, canvas_size))

        # 4. 横向拼接
        final_img = np.hstack((rgb_resized, depth_resized, traj_resized))

        # 5. 保存日志文本信息到txt
        log_txt = f"goal: {log_goal}\n"
        log_txt += f"traj: {log_traj}\n"
        log_txt += f"reward: {log_traj_reward}\n"
        log_txt += f"reward_info: {log_traj_reward_info}\n"
        
        return final_img, log_txt