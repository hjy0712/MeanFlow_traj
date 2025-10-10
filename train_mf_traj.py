import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from accelerate import Accelerator
import matplotlib.pyplot as plt
import numpy as np

from models_traj.policy_network_meanflow import NavDP_Policy_MeanFlow
from dataset.dataset3dfront import NavDP_Base_Datset, navdp_collate_fn


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def main():
    # ---------- config ----------
    n_steps = 200000
    batch_size = 64
    log_step = 20
    sample_step = 100
    ckpt_step = 2000
    lr = 1e-4

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    if accelerator.is_main_process:
        print(f"Using device: {device}, world size={accelerator.num_processes}")

    os.makedirs("runs/images", exist_ok=True)
    os.makedirs("runs/checkpoints", exist_ok=True)

    # ---------- TensorBoard init ----------
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir="runs/tensorboard_logs")

    # ---------- dataset ----------
    # base_paths = [
    #     "/mnt/zrh/data/static_nav_from_n1/3dfront_zed",
    #     "/mnt/zrh/data/static_nav_from_n1/3dfront_d435i"
    # ]
    base_paths = "/mnt/zrh/data/static_nav_from_n1/3dfront_zed"
    dataset = NavDP_Base_Datset(
        root_dirs=base_paths,
        memory_size=8,
        predict_size=24,
        image_size=224,
        scene_data_scale=1.0,
        trajectory_data_scale=1.0,
        preload=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        collate_fn=navdp_collate_fn,
    )
    dataloader = cycle(dataloader)

    # ---------- model ----------
    model = NavDP_Policy_MeanFlow(
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=8,
        heads=8,
        token_dim=384,
        device=device,
        solver_steps=50,
    )
    net = unwrap_model(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    # prepare for DDP
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    global_step = 0
    running_loss = 0.0

    # ---------- training loop ----------
    with tqdm(range(n_steps), disable=not accelerator.is_main_process, dynamic_ncols=True) as pbar:
        pbar.set_description("Training")
        model.train()

        for _ in pbar:
            batch = next(dataloader)

            # ---- inputs from dataset ----
            traj_target = batch["batch_labels"].to(device)
            input_images = batch["batch_rgb"].to(device)
            input_depths = batch["batch_depth"].to(device)
            goal_point = batch["batch_pg"].to(device)

            # ---- embeddings ----
            rgbd_embed = net.rgbd_encoder(input_images, input_depths)
            pointgoal_embed = net.point_encoder(goal_point).unsqueeze(1)

            # ---- a_start: 高斯轨迹 ----
            a_start = torch.randn_like(traj_target)

            # # ---- flow matching loss ----
            # loss = net.compute_flow_matching_loss(
            #     a_start=a_start,
            #     a_target=traj_target,
            #     goal=goal_point,
            #     goal_embed=pointgoal_embed,
            #     rgbd_embed=rgbd_embed,
            # )

            # ---- meanflow loss ----
            loss, mse_val = net.compute_meanflow_loss(
                model_fn=net.predict_velocity,   # 主体模型函数
                x=traj_target,                   # 输入轨迹（ground truth）
                goal_embed=pointgoal_embed,      # 目标点 embedding
                rgbd_embed=rgbd_embed            # 感知 embedding
            )

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            running_loss += loss.item()

            # ---------- log ----------
            if accelerator.is_main_process and global_step % log_step == 0:
                avg_loss = running_loss / log_step
                current_time = time.asctime(time.localtime(time.time()))
                log_msg = f"{current_time} | step={global_step} | loss={avg_loss:.6f}"
                print(log_msg)
                with open("runs/train_log.txt", "a") as f:
                    f.write(log_msg + "\n")

                writer.add_scalar("Loss/train", avg_loss, global_step)
                running_loss = 0.0

            # ---------- save checkpoint ----------
            if accelerator.is_main_process and global_step % ckpt_step == 0:
                ckpt_path = f"runs/checkpoints/navdpflow_step_{global_step}.pt"
                accelerator.save(model.state_dict(), ckpt_path)

            # ---------- sample ----------
            if accelerator.is_main_process and global_step % sample_step == 0:
                model.eval()
                with torch.no_grad():
                    sample_pic = test_in_train(1, dataloader, device, model, global_step)
                    if sample_pic is not None:
                        img_save_path = f"runs/images/step_{global_step}.png"
                        sample_pic.savefig(img_save_path, dpi=150, bbox_inches='tight')
                        print(f"Sample image saved at step {global_step} to {img_save_path}")
                        
                model.train()

    # ---------- final save checkpoint ----------
    if accelerator.is_main_process:
        ckpt_path = f"runs/checkpoints/navdpflow_step_{global_step}.pt"
        accelerator.save(model.state_dict(), ckpt_path)
        writer.close()

def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

def test_in_train(sample_batch_size, dataloader, device, model, global_step):
    net = unwrap_model(model)
    
    # 随机获取一个批次的数据
    sample_batch = next(dataloader)
    
    # 获取采样数据
    sample_traj_target = sample_batch["batch_labels"].to(device)[:sample_batch_size]
    sample_input_images = sample_batch["batch_rgb"].to(device)[:sample_batch_size]
    sample_input_depths = sample_batch["batch_depth"].to(device)[:sample_batch_size]
    sample_goal_point = sample_batch["batch_pg"].to(device)[:sample_batch_size]
    
    # 模型推理轨迹
    trajs, critics, pos_traj, neg_traj = net.predict_pointgoal_action(
        goal_point=sample_goal_point,
        input_images=sample_input_images,
        input_depths=sample_input_depths,
        sample_num=8,
    )
    
    # 创建图像
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 绘制数据集监督轨迹（红色）
    for i in range(sample_batch_size):
        target_traj = sample_traj_target[i].cpu().numpy()
        # 计算累积位置（从起点开始）
        cumulative_pos = np.cumsum(target_traj/4.0, axis=0)
        # 添加起点 (0, 0)
        full_traj = np.vstack([[0, 0, 0], cumulative_pos])
        
        # 绘制轨迹线
        ax.plot(full_traj[:, 0], full_traj[:, 1], 'r-', linewidth=2, alpha=0.7, label='Ground Truth' if i == 0 else "")
        # 用圆圈标记每个轨迹点
        ax.scatter(full_traj[:, 0], full_traj[:, 1], color='red', s=50, marker='o', alpha=0.8, label='GT Points' if i == 0 else "")
        # 标记起点和终点
        ax.scatter(full_traj[0, 0], full_traj[0, 1], color='green', s=100, marker='o', label='Start' if i == 0 else "")
        ax.scatter(full_traj[-1, 0], full_traj[-1, 1], color='red', s=100, marker='s', label='GT End' if i == 0 else "")
    
    # 绘制模型推理轨迹（黑色）- 绘制所有轨迹
    for i in range(sample_batch_size):
        # 绘制所有生成的轨迹
        for j in range(trajs.shape[1]):  # trajs.shape[1] 是轨迹数量
            pred_traj = trajs[i, j]
            # 添加起点 (0, 0)
            full_traj = np.vstack([[0, 0, 0], pred_traj])
            
            # 绘制轨迹线（使用不同的透明度来区分多条轨迹）
            alpha = 0.5 if j > 0 else 0.7  # 第一条轨迹更明显
            ax.plot(full_traj[:, 0], full_traj[:, 1], 'k-', linewidth=1.5, alpha=alpha, 
                   label='Predicted Trajectories' if i == 0 and j == 0 else "")
            # 用圆圈标记每个轨迹点
            ax.scatter(full_traj[:, 0], full_traj[:, 1], color='black', s=30, marker='o', alpha=alpha*0.8, 
                      label='Pred Points' if i == 0 and j == 0 else "")
            # 标记终点
            ax.scatter(full_traj[-1, 0], full_traj[-1, 1], color='black', s=60, marker='^', alpha=alpha, 
                      label='Pred End' if i == 0 and j == 0 else "")
    
    # 绘制期望目标点
    for i in range(sample_batch_size):
        goal_point = sample_goal_point[i].cpu().numpy()
        ax.scatter(goal_point[0], goal_point[1], color='blue', s=150, marker='*', label='Target Goal' if i == 0 else "")
    
    # 设置图像属性
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Trajectory Comparison - Step {global_step}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    
    # 保存图像
    plt.tight_layout()
    sample_pic = fig
    plt.close(fig)  # 关闭图形以释放内存
    
    return sample_pic

if __name__ == "__main__":
    main()