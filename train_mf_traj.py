import os
import time
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
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


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def main():
    # ---------- config ----------
    n_steps = 50000
    epoch = 4
    batch_size = 512
    log_step = 20
    sample_step = 100
    ckpt_step = 2000
    lr = 1e-4

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    print(f"[Rank {accelerator.process_index}] Using device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
    if accelerator.is_main_process: 
        print(f"Using {accelerator.num_processes} processes across {torch.cuda.device_count()} GPUs")

    os.makedirs("runstest/images", exist_ok=True)
    os.makedirs("runstest/checkpoints", exist_ok=True)

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir="runstest/tensorboard_logs")

    # ---------- dataset ----------
    # base_paths = [
    #     "/mnt/zrh/data/static_nav_from_n1/3dfront_zed",
    #     "/mnt/zrh/data/static_nav_from_n1/3dfront_d435i"
    # ]
    base_paths = "/mnt/zrh/data/static_nav_from_n1/3dfront_d435i"
    dataset = NavDP_Base_Datset(
        root_dirs=base_paths,
        memory_size=8,
        predict_size=24,
        image_size=224,
        scene_data_scale=1.0,
        trajectory_data_scale=1.0,
        debug=True,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-6)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    net = unwrap_model(model)

    global_step = 0
    running_loss = 0.0

    # 每个 epoch 重复次数（每个 batch 更新几次）
    repeat_per_batch = epoch

    # ---------- training loop ----------
    with tqdm(range(n_steps), disable=not accelerator.is_main_process, dynamic_ncols=True) as pbar:
        pbar.set_description("Training")
        model.train()

        for step in pbar:
            batch = next(dataloader)
            traj_target = batch["batch_labels"].to(device)
            input_images = batch["batch_rgb"].to(device)
            input_depths = batch["batch_depth"].to(device)
            goal_point = batch["batch_pg"].to(device)

            # 重复更新 repeat_per_batch 次
            for _ in range(repeat_per_batch):
                print(input_images.shape, input_depths.shape, goal_point.shape)
                rgbd_embed = net.rgbd_encoder(input_images, input_depths)
                pointgoal_embed = net.point_encoder(goal_point).unsqueeze(1)
                a_start = torch.randn_like(traj_target)

                loss, mse_val = net.compute_meanflow_loss(
                    model_fn=net.predict_velocity,
                    a_start=a_start,
                    a_target=traj_target,
                    goal_embed=pointgoal_embed,
                    rgbd_embed=rgbd_embed
                )

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                running_loss += loss.item()

                # ---------- log ----------
                if accelerator.is_main_process and global_step % log_step == 0:
                    avg_loss = running_loss / log_step
                    current_lr = scheduler.get_last_lr()[0]
                    current_time = time.asctime(time.localtime(time.time()))
                    log_msg = f"{current_time} | step={global_step} | loss={avg_loss:.6f}"
                    print(log_msg)
                    with open("runs/train_log.txt", "a") as f:
                        f.write(log_msg + "\n")

                    writer.add_scalar("Loss/train", avg_loss, global_step)
                    writer.add_scalar("LR", current_lr, global_step)
                    running_loss = 0.0

                # ---------- checkpoint ----------
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

                if global_step >= n_steps:
                    break
            if global_step >= n_steps:
                break

    # ---------- final save checkpoint ----------
    if accelerator.is_main_process:
        ckpt_path = f"runs/checkpoints/navdpflow_step_{global_step}.pt"
        accelerator.save(model.state_dict(), ckpt_path)
        writer.close()


def test_in_train(sample_batch_size, dataloader, device, model, global_step):
    net = unwrap_model(model)
    sample_batch = next(dataloader)

    sample_traj_target = sample_batch["batch_labels"].to(device)[:sample_batch_size]
    sample_input_images = sample_batch["batch_rgb"].to(device)[:sample_batch_size]
    sample_input_depths = sample_batch["batch_depth"].to(device)[:sample_batch_size]
    sample_goal_point = sample_batch["batch_pg"].to(device)[:sample_batch_size]

    trajs, critics, pos_traj, neg_traj = net.predict_pointgoal_action(
        goal_point=sample_goal_point,
        input_images=sample_input_images,
        input_depths=sample_input_depths,
        sample_num=8,
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for i in range(sample_batch_size):
        target_traj = sample_traj_target[i].cpu().numpy()
        cumulative_pos = np.cumsum(target_traj/4.0, axis=0)
        full_traj = np.vstack([[0, 0, 0], cumulative_pos])
        ax.plot(full_traj[:, 0], full_traj[:, 1], 'r-', linewidth=2, alpha=0.7)
        ax.scatter(full_traj[:, 0], full_traj[:, 1], color='red', s=50, marker='o', alpha=0.8)
        ax.scatter(full_traj[0, 0], full_traj[0, 1], color='green', s=100, marker='o')
        ax.scatter(full_traj[-1, 0], full_traj[-1, 1], color='red', s=100, marker='s')

    for i in range(sample_batch_size):
        for j in range(trajs.shape[1]):
            pred_traj = trajs[i, j]
            full_traj = np.vstack([[0, 0, 0], pred_traj])
            alpha = 0.5 if j > 0 else 0.7
            ax.plot(full_traj[:, 0], full_traj[:, 1], 'k-', linewidth=1.5, alpha=alpha)
            ax.scatter(full_traj[:, 0], full_traj[:, 1], color='black', s=30, marker='o', alpha=alpha*0.8)
            ax.scatter(full_traj[-1, 0], full_traj[-1, 1], color='black', s=60, marker='^', alpha=alpha)

    for i in range(sample_batch_size):
        goal_point = sample_goal_point[i].cpu().numpy()
        ax.scatter(goal_point[0], goal_point[1], color='blue', s=150, marker='*')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Trajectory Comparison - Step {global_step}')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()
    sample_pic = fig
    plt.close(fig)
    return sample_pic


if __name__ == "__main__":
    main()
