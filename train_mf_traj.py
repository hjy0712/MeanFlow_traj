import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

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
    log_step = 200
    sample_step = 2000
    lr = 1e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("runs/images", exist_ok=True)
    os.makedirs("runs/checkpoints", exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16")

    # ---------- dataset ----------
    dataset = NavDP_Base_Datset(
        root_dirs="/mnt/zrh/data/static_nav_from_n1/3dfront_zed",
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
        num_workers=8,
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
                running_loss = 0.0

            # ---------- sample ----------
            if accelerator.is_main_process and global_step % sample_step == 0:
                model.eval()
                with torch.no_grad():
                    trajs, critics, pos_traj, neg_traj = model.predict_pointgoal_action(
                        goal_point=goal_point[:2],
                        input_images=input_images[:2],
                        input_depths=input_depths[:2],
                        sample_num=8,
                    )
                    import numpy as np
                    np.save(f"runs/images/sample_step_{global_step}.npy", trajs)
                model.train()

    # ---------- save checkpoint ----------
    if accelerator.is_main_process:
        ckpt_path = f"runs/checkpoints/navdpflow_step_{global_step}.pt"
        accelerator.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
