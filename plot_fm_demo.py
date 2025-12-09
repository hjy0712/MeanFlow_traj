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


def plot_trajectory(ax, gt_traj, pred_trajs, goal_point, title="", show_legend=False):
    """
    在给定的axes上绘制轨迹图
    坐标轴方向：x轴正方向竖直向上，y轴正方向水平向左
    
    Args:
        ax: matplotlib axes对象
        gt_traj: 真实轨迹
        pred_trajs: 预测轨迹列表
        goal_point: 目标点
        title: 标题
        show_legend: 是否显示图例，默认False
    """
    # Ground Truth
    full_gt = np.vstack([[0, 0, 0], np.cumsum(gt_traj / 4.0, axis=0)])
    # x轴正方向竖直向上：使用x坐标作为y轴（向上为正，默认）
    # y轴正方向水平向左：使用y坐标作为x轴（向左为正，需要反转x轴）
    ax.plot(full_gt[:, 1], full_gt[:, 0], 'r-', linewidth=3, label="GT Trajectory" if show_legend else "")
    ax.scatter(full_gt[:, 1], full_gt[:, 0], color="red", s=40, alpha=0.6)
    ax.scatter(0, 0, color="green", s=100, label="Start" if show_legend else "", zorder=5)
    ax.scatter(full_gt[-1, 1], full_gt[-1, 0], color="red", s=100, marker="s", label="GT End" if show_legend else "", zorder=5)

    # Predicted
    for i, traj in enumerate(pred_trajs):
        full_pred = np.vstack([[0, 0, 0], traj])
        ax.plot(full_pred[:, 1], full_pred[:, 0], 'b-', alpha=0.5, linewidth=2, 
                label="Pred Traj" if (show_legend and i == 0) else "")
        ax.scatter(full_pred[-1, 1], full_pred[-1, 0], color="blue", s=60, marker="^", alpha=0.7)

    # Goal
    ax.scatter(goal_point[1], goal_point[0], color="orange", s=180, marker="*", 
              label="Goal" if show_legend else "", zorder=5, edgecolors='black', linewidths=0.5)

    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel("Y (Left ←)", fontsize=16, fontweight='bold')
    ax.set_ylabel("X (Front →)", fontsize=16, fontweight='bold')
    # 反转x轴（y轴正方向向左）
    ax.invert_xaxis()
    if show_legend and len(pred_trajs) > 0:
        ax.legend(fontsize=14, loc='upper right')
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    # 增大刻度字体
    ax.tick_params(labelsize=14)


def save_results_figure(rgb, depth, gt_traj, pred_trajs_dict, goal_point, save_dir, step):
    """
    保存 RGB + Depth + 四个不同去噪步数的轨迹图
    pred_trajs_dict: {solver_steps: pred_trajs} 字典，键为去噪步数，值为预测轨迹列表
    """
    os.makedirs(save_dir, exist_ok=True)

    # 创建布局：上方2个图（RGB和Depth），下方4个轨迹图
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, 
                         left=0.05, right=0.95, top=0.95, bottom=0.05)

    # ---- RGB (去除黑边) ----
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_rgb.imshow(rgb)
    ax_rgb.set_title("RGB", fontsize=20, fontweight='bold', pad=15)
    ax_rgb.axis("off")
    # 去除黑边：设置边距为0
    ax_rgb.margins(x=0, y=0)

    # ---- Depth (去除黑边) ----
    ax_depth = fig.add_subplot(gs[0, 1])
    im = ax_depth.imshow(depth, cmap="gray")
    ax_depth.set_title("Depth", fontsize=20, fontweight='bold', pad=15)
    ax_depth.axis("off")
    ax_depth.margins(x=0, y=0)
    cbar = fig.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)

    # 移除第三个位置（空出来）
    ax_empty = fig.add_subplot(gs[0, 2])
    ax_empty.axis("off")

    # ---- 四个不同去噪步数的轨迹图 ----
    solver_steps_list = [1, 3, 5, 10]
    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
    
    for idx, (solver_steps, pos) in enumerate(zip(solver_steps_list, positions)):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        if solver_steps in pred_trajs_dict:
            pred_trajs = pred_trajs_dict[solver_steps]
            # 只有最后一个图（idx == 3，即去噪步数40）显示图例
            show_legend = (idx == len(solver_steps_list) - 1)
            plot_trajectory(ax, gt_traj, pred_trajs, goal_point, 
                          title=f"Denoising Steps: {solver_steps}",
                          show_legend=show_legend)
        else:
            ax.text(0.5, 0.5, f"No data for {solver_steps} steps", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f"Denoising Steps: {solver_steps}", fontsize=18, fontweight='bold')
            ax.axis("off")

    # ---- Save ----
    save_path = os.path.join(save_dir, f"result_step{step}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
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
    save_root = "tests/results_denoising_steps"
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
    solver_steps_list = [1, 3, 5, 10]  # 不同的去噪步数（需要与save_results_figure中的列表保持一致）
    
    for step, batch in enumerate(tqdm(dataloader, desc="Testing")):
        traj_target = batch["batch_labels"].to(device)   # (1,T,3)
        input_images = batch["batch_rgb"].to(device)     # (1,mem,H,W,3)
        input_depths = batch["batch_depth"].to(device)   # (1,mem,H,W,1)
        goal_point = batch["batch_pg"].to(device)        # (1,3)

        # numpy 化（这些不依赖于去噪步数）
        gt_traj = traj_target[0].cpu().numpy()
        goal = goal_point[0].cpu().numpy()
        rgb = input_images[0, -1].cpu().numpy()
        depth = input_depths[0].cpu().numpy().squeeze()

        # 为每个去噪步数生成轨迹
        pred_trajs_dict = {}
        for solver_steps in solver_steps_list:
            # 临时修改模型的solver_steps
            original_steps = model.solver_steps
            model.solver_steps = solver_steps
            
            with torch.no_grad():
                trajs, _, _, _ = model.predict_pointgoal_action(
                    goal_point=goal_point,
                    input_images=input_images,
                    input_depths=input_depths,
                    sample_num=8,
                )
            pred_trajs_dict[solver_steps] = trajs[0]
            
            # 恢复原始solver_steps
            model.solver_steps = original_steps

        # 保存结果图（包含RGB、Depth和四个轨迹图）
        save_results_figure(rgb, depth, gt_traj, pred_trajs_dict, goal, 
                          os.path.join(save_root, "results"), step)

        # 另外保存txt（使用第一个去噪步数的结果）
        save_traj_txt(gt_traj, pred_trajs_dict[solver_steps_list[0]], goal, 
                     os.path.join(save_root, "txt"), step)


if __name__ == "__main__":
    main()
