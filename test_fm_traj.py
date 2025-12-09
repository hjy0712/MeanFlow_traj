"""
加载训练好的checkpoint，测试效果并保存观测rgbd图和轨迹图 + 轨迹txt
"""
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models_traj.policy_network import NavDP_Policy_Flow
from dataset.dataset3dfront import NavDP_Base_Datset, navdp_collate_fn


def save_results_figure(rgb, depth, gt_traj, pred_trajs, goal_point, save_dir, step):
    """
    保存 RGB + Depth + Trajectory 三合一结果图
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ---- RGB ----
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    # ---- Depth ----
    im = axes[1].imshow(depth, cmap="gray")
    axes[1].set_title("Depth")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # ---- Trajectory ----
    ax = axes[2]
    # Ground Truth
    full_gt = np.vstack([[0, 0, 0], np.cumsum(gt_traj / 4.0, axis=0)])
    ax.plot(full_gt[:, 0], full_gt[:, 1], 'r-', linewidth=2, label="GT Trajectory")
    ax.scatter(full_gt[:, 0], full_gt[:, 1], color="red", s=30)
    ax.scatter(0, 0, color="green", s=80, label="Start")
    ax.scatter(full_gt[-1, 0], full_gt[-1, 1], color="red", s=80, marker="s", label="GT End")

    # Predicted
    for i, traj in enumerate(pred_trajs):
        full_pred = np.vstack([[0, 0, 0], traj])
        ax.plot(full_pred[:, 0], full_pred[:, 1], 'k-', alpha=0.6, label="Pred Traj" if i == 0 else "")
        ax.scatter(full_pred[-1, 0], full_pred[-1, 1], color="black", s=50, marker="^")

    # Goal
    ax.scatter(goal_point[0], goal_point[1], color="blue", s=150, marker="*", label="Goal")

    ax.set_title("Trajectory Comparison")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    # ---- Save ----
    save_path = os.path.join(save_dir, f"result_step{step}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path}")


def save_traj_txt(gt_traj, pred_trajs, goal_point, save_dir, step):
    """
    保存GT和Pred轨迹为txt
    """
    os.makedirs(save_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, f"traj_step{step}.txt")

    with open(txt_path, "w") as f:
        f.write("Goal Point: " + " ".join(map(str, goal_point.tolist())) + "\n")
        f.write("Ground Truth Trajectory:\n")
        np.savetxt(f, gt_traj, fmt="%.6f")
        f.write("\nPredicted Trajectories:\n")
        for i, traj in enumerate(pred_trajs):
            f.write(f"Traj {i}:\n")
            np.savetxt(f, traj, fmt="%.6f")
            f.write("\n")


def main():
    # ---------- config ----------
    ckpt_path = "/mnt/houjunyi/MeanFlow_traj/runs_fm/navdpflow_step_149000.pt"
    save_root = "tests/results_3"
    os.makedirs(save_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---------- dataset ----------
    # base_paths = [
    #     "/mnt/zrh/data/static_nav_from_n1/3dfront_zed",
    #     # "/mnt/zrh/data/static_nav_from_n1/3dfront_d435i"
    # ]
    base_paths = "/mnt/zrh/data/static_nav_from_n1/3dfront_zed"
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
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=navdp_collate_fn,
    )

    # ---------- model ----------
    model = NavDP_Policy_Flow(
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=8,
        heads=8,
        token_dim=384,
        device=device,
        solver_steps=3,
    ).to(device)

    # 加载 checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    # # 如果是单卡训练的，直接加载
    # model.load_state_dict(ckpt)
    # 如果是多卡保存的，加这段
    new_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(new_ckpt, strict=False)

    model.eval()

    # ---------- inference ----------
    for step, batch in enumerate(tqdm(dataloader, desc="Testing")):
        traj_target = batch["batch_labels"].to(device)   # (1,T,3)
        input_images = batch["batch_rgb"].to(device)     # (1,mem,H,W,3)
        input_depths = batch["batch_depth"].to(device)   # (1,mem,H,W,1)
        goal_point = batch["batch_pg"].to(device)        # (1,3)

        with torch.no_grad():
            import ipdb; ipdb.set_trace()
            trajs, _, _, _ = model.predict_pointgoal_action(
                goal_point=goal_point,
                input_images=input_images,
                input_depths=input_depths,
                sample_num=8,
            )
        # numpy 化
        gt_traj = traj_target[0].cpu().numpy()
        pred_trajs = trajs[0]
        goal = goal_point[0].cpu().numpy()
        rgb = input_images[0, -1].cpu().numpy()
        depth = input_depths[0].cpu().numpy().squeeze()

        # 保存三合一图
        save_results_figure(rgb, depth, gt_traj, pred_trajs, goal, os.path.join(save_root, "results"), step)

        # 另外保存txt
        save_traj_txt(gt_traj, pred_trajs, goal, os.path.join(save_root, "txt"), step)


if __name__ == "__main__":
    main()
