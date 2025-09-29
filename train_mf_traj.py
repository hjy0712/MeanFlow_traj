import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from accelerate import Accelerator
import matplotlib.pyplot as plt
import numpy as np

from models_traj.policy_network import NavDP_Policy_Flow
from dataset.dataset3dfront import NavDP_Base_Datset, navdp_collate_fn


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def main():
    # ---------- config ----------
    n_steps = 200000
    batch_size = 16
    log_step = 100
    sample_step = 100
    ckpt_step = 500
    lr = 1e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("runs/images", exist_ok=True)
    os.makedirs("runs/checkpoints", exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16")

    # ---------- TensorBoard init ----------
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir="runs/tensorboard_logs")

    # ---------- dataset ----------
    base_paths = [
        "/mnt/zrh/data/static_nav_from_n1/3dfront_zed",  # 第一个基础路径
        "/mnt/zrh/data/static_nav_from_n1/3dfront_d435i"  # 第二个基础路径（请替换为实际路径）
    ]
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
        num_workers=0,
        collate_fn=navdp_collate_fn,
    )
    dataloader = cycle(dataloader)

    # ---------- model ----------
    model = NavDP_Policy_Flow(
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=8,
        heads=8,
        token_dim=384,
        device=device,
        solver_steps=50,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    global_step = 0
    running_loss = 0.0

    # ---------- training loop ----------
    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training")
        model.train()

        for _ in pbar:
            batch = next(dataloader)

            # ---- inputs from dataset ----
            traj_target = batch["batch_labels"].to(device)   # (B,T,3)
            input_images = batch["batch_rgb"].to(device)     # (B,mem,H,W,3)
            input_depths = batch["batch_depth"].to(device)   # (B,mem,H,W,1)
            goal_point = batch["batch_pg"].to(device)        # (B,3)

            # ---- embeddings ----
            rgbd_embed = model.rgbd_encoder(input_images, input_depths)
            pointgoal_embed = model.point_encoder(goal_point).unsqueeze(1)

            # ---- a_start: 全零轨迹 ----
            a_start = torch.zeros_like(traj_target)

            # ---- flow matching loss ----
            loss = model.compute_flow_matching_loss(
                a_start=a_start,
                a_target=traj_target,
                goal=goal_point,
                goal_embed=pointgoal_embed,
                rgbd_embed=rgbd_embed,
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

                # TensorBoard log
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
                    # 随机选取一个场景进行采样
                    sample_batch_size = 1
                    sample_pic = test_in_train(sample_batch_size, dataloader, device, model, global_step)
                    # 保存采样图片
                    if sample_pic is not None:
                        os.makedirs("runs/images", exist_ok=True)
                        img_save_path = f"runs/images/step_{global_step}.png"
                        sample_pic.savefig(img_save_path, dpi=150, bbox_inches='tight')
                        print(f"Sample image saved at step {global_step} to {img_save_path}")
                        
                model.train()

    # ---------- final save checkpoint ----------
    if accelerator.is_main_process:
        ckpt_path = f"runs/checkpoints/navdpflow_step_{global_step}.pt"
        accelerator.save(model.state_dict(), ckpt_path)
        writer.close()

def test_in_train(sample_batch_size, dataloader, device, model, global_step):
    
    # 随机获取一个批次的数据
    sample_batch = next(dataloader)
    
    # 获取采样数据
    sample_traj_target = sample_batch["batch_labels"].to(device)[:sample_batch_size]
    sample_input_images = sample_batch["batch_rgb"].to(device)[:sample_batch_size]
    sample_input_depths = sample_batch["batch_depth"].to(device)[:sample_batch_size]
    sample_goal_point = sample_batch["batch_pg"].to(device)[:sample_batch_size]
    
    # 模型推理轨迹
    trajs, critics, pos_traj, neg_traj = model.predict_pointgoal_action(
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
        # 添加起点 (0, 0)
        full_traj = np.vstack([[0, 0, 0], target_traj])
        
        # 绘制轨迹
        ax.plot(full_traj[:, 0], full_traj[:, 1], 'r-', linewidth=2, alpha=0.7, label='Ground Truth' if i == 0 else "")
        # 标记起点和终点
        ax.scatter(full_traj[0, 0], full_traj[0, 1], color='green', s=100, marker='o', label='Start' if i == 0 else "")
        ax.scatter(full_traj[-1, 0], full_traj[-1, 1], color='red', s=100, marker='s', label='GT End' if i == 0 else "")
    
    # 绘制模型推理轨迹（黑色）
    for i in range(sample_batch_size):
        # 选择critic值最高的轨迹
        best_idx = np.argmax(critics[i])
        pred_traj = trajs[i, best_idx]
        
        # 计算累积位置
        cumulative_pos = np.cumsum(pred_traj, axis=0)
        # 添加起点 (0, 0)
        full_traj = np.vstack([[0, 0, 0], cumulative_pos])
        
        # 绘制轨迹
        ax.plot(full_traj[:, 0], full_traj[:, 1], 'k-', linewidth=2, alpha=0.7, label='Predicted' if i == 0 else "")
        # 标记终点
        ax.scatter(full_traj[-1, 0], full_traj[-1, 1], color='black', s=100, marker='^', label='Pred End' if i == 0 else "")
    
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
