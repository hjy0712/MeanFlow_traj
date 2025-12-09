"""
加载训练好的三个checkpoint，测试单步效果并保存观测rgbd图和轨迹图 + 轨迹txt
"""
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from tqdm import tqdm

from models_traj.policy_network import NavDP_Policy_Flow
from models_traj.policy_network_meanflow import NavDP_Policy_MeanFlow
from dataset.dataset3dfront import NavDP_Base_Datset, navdp_collate_fn


def plot_trajectory(ax, gt_traj, pred_trajs, goal_point, title="", show_legend=False, pred_color='blue', pred_label="Pred Traj", set_limits=None):
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
        pred_color: 预测轨迹的颜色
        pred_label: 预测轨迹的标签
        set_limits: 如果提供，格式为(xlim, ylim)，设置固定的坐标轴范围；如果None，则自适应
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
        ax.plot(full_pred[:, 1], full_pred[:, 0], color=pred_color, alpha=0.5, linewidth=2, 
                label=pred_label if (show_legend and i == 0) else "")
        ax.scatter(full_pred[-1, 1], full_pred[-1, 0], color=pred_color, s=60, marker="^", alpha=0.7)

    # Goal
    ax.scatter(goal_point[1], goal_point[0], color="orange", s=180, marker="*", 
              label="Goal" if show_legend else "", zorder=5, edgecolors='black', linewidths=0.5)

    # 移除小标题
    # ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel("Y (Left ←)", fontsize=20, fontweight='bold')  # 增大坐标轴标签字体
    ax.set_ylabel("X (Front →)", fontsize=20, fontweight='bold')  # 增大坐标轴标签字体
    # 不再在子图中显示图例，统一在右下角显示
    # 设置子图比例为2:1（高度:宽度=2:1，即瘦高）
    # 使用set_box_aspect来设置子图的宽高比（高度:宽度=2:1）
    try:
        ax.set_box_aspect(2.0)  # 高度是宽度的2倍（matplotlib 3.5+）
    except AttributeError:
        # 如果matplotlib版本较旧，使用set_aspect
        ax.set_aspect('auto')
        # 通过调整坐标轴范围来近似实现2:1比例
        pass
    # 设置坐标轴范围（在反转x轴之前设置，确保范围正确）
    if set_limits is not None:
        xlim, ylim = set_limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    # 否则自适应（不设置范围）
    
    # 反转x轴（y轴正方向向左）- 在所有设置之后反转，确保方向一致
    ax.invert_xaxis()
    
    # 移除所有边框（黑色框）
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.grid(True, alpha=0.3)
    # 增大刻度字体
    ax.tick_params(labelsize=22)  # 增大坐标轴数字字体大小


def calculate_trajectory_limits(gt_traj, pred_trajs_list, goal_point):
    """
    计算轨迹数据的坐标轴范围
    
    Args:
        gt_traj: 真实轨迹
        pred_trajs_list: 预测轨迹列表的列表（多个子图的轨迹）
        goal_point: 目标点
    
    Returns:
        (xlim, ylim): 坐标轴范围元组
    """
    all_x = []
    all_y = []
    
    # GT轨迹
    full_gt = np.vstack([[0, 0, 0], np.cumsum(gt_traj / 4.0, axis=0)])
    all_x.extend(full_gt[:, 0])
    all_y.extend(full_gt[:, 1])
    
    # 所有预测轨迹
    for pred_trajs in pred_trajs_list:
        for traj in pred_trajs:
            full_pred = np.vstack([[0, 0, 0], traj])
            all_x.extend(full_pred[:, 0])
            all_y.extend(full_pred[:, 1])
    
    # 目标点
    all_x.append(goal_point[0])
    all_y.append(goal_point[1])
    
    # 起点
    all_x.append(0)
    all_y.append(0)
    
    # 计算范围，添加一些边距
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # 添加10%的边距
    x_margin = x_range * 0.1
    y_margin = y_range * 0.1
    
    xlim = (x_min - x_margin, x_max + x_margin)
    ylim = (y_min - y_margin, y_max + y_margin)
    
    return (xlim, ylim)


def save_results_figure(rgb, depth, gt_traj, pred_trajs_dict, goal_point, save_dir, step):
    """
    保存 RGB + Depth + Flow Matching不同步数 + Mean Flow的轨迹图
    pred_trajs_dict: {key: pred_trajs} 字典
        - "runs_fm_1", "runs_fm_3", "runs_fm_10": Flow Matching不同步数
        - "runs_mf0.8", "runs_mf0.5": Mean Flow不同配置
    """
    os.makedirs(save_dir, exist_ok=True)

    # 创建布局：上方2个图（RGB和Depth），下方5个轨迹图
    # 调整figure大小，使轨迹图比例为2:1（瘦高）
    fig = plt.figure(figsize=(18, 20))
    # 使用width_ratios和height_ratios来控制子图比例
    # 轨迹图需要2:1（高度:宽度=2:1），所以行高应该是列宽的2倍
    # 调整列宽比例，使轨迹图更瘦高
    gs = fig.add_gridspec(3, 3, 
                         hspace=0.35, wspace=0.3, 
                         height_ratios=[1, 2, 2],  # 第一行（RGB/Depth）高度为1，轨迹图行高度为2
                         width_ratios=[1, 1, 1],   # 列宽相等
                         left=0.05, right=0.95, top=0.95, bottom=0.05)

    # ---- RGB (去除黑边) ----
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_rgb.imshow(rgb)
    # 移除标题
    # ax_rgb.set_title("RGB", fontsize=20, fontweight='bold', pad=15)
    ax_rgb.axis("off")
    ax_rgb.margins(x=0, y=0)

    # ---- Depth (去除黑边) ----
    ax_depth = fig.add_subplot(gs[0, 1])
    im = ax_depth.imshow(depth, cmap="gray")
    # 移除标题
    # ax_depth.set_title("Depth", fontsize=20, fontweight='bold', pad=15)
    ax_depth.axis("off")
    ax_depth.margins(x=0, y=0)
    cbar = fig.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)

    # 移除第三个位置（空出来）
    ax_empty = fig.add_subplot(gs[0, 2])
    ax_empty.axis("off")

    # ---- 计算其他四个子图的统一坐标轴范围 ----
    # 需要统一范围的子图：FM 3步、10步，MF 0.8、0.5
    unified_keys = ["runs_fm_3", "runs_fm_10", "runs_mf0.8", "runs_mf0.5"]
    unified_pred_trajs_list = []
    for key in unified_keys:
        if key in pred_trajs_dict:
            unified_pred_trajs_list.append(pred_trajs_dict[key])
    
    # 计算统一范围
    if unified_pred_trajs_list:
        unified_limits = calculate_trajectory_limits(gt_traj, unified_pred_trajs_list, goal_point)
        # 注意：在plot_trajectory中，ax.plot使用的是 (Y坐标, X坐标)
        # xlim对应Y坐标（横轴），ylim对应X坐标（竖轴）
        # calculate_trajectory_limits返回的是 (X坐标范围, Y坐标范围)
        unified_xlim = unified_limits[1]  # Y坐标范围（横轴，对应plot中的第一个参数）
        unified_ylim = unified_limits[0]  # X坐标范围（竖轴，对应plot中的第二个参数）
    else:
        unified_xlim = None
        unified_ylim = None

    # ---- Flow Matching不同步数的轨迹图 ----
    fm_configs = [
        ("runs_fm_1", "Flow Matching (1 step)", '#2980B9', (1, 0), None),  # 1步自适应
        ("runs_fm_3", "Flow Matching (3 steps)", '#3498DB', (1, 1), (unified_xlim, unified_ylim)),  # 使用统一范围
        ("runs_fm_10", "Flow Matching (10 steps)", '#5DADE2', (1, 2), (unified_xlim, unified_ylim)),  # 使用统一范围
    ]
    
    for key, name, color, pos, limits in fm_configs:
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        if key in pred_trajs_dict:
            pred_trajs = pred_trajs_dict[key]
            plot_trajectory(ax, gt_traj, pred_trajs, goal_point, 
                          title=name,
                          show_legend=False,
                          pred_color=color,
                          pred_label=name,
                          set_limits=limits)  # 传入坐标轴范围
        else:
            ax.text(0.5, 0.5, f"No data for {name}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            # 移除标题
            # ax.set_title(name, fontsize=18, fontweight='bold')
            ax.axis("off")

    # ---- Mean Flow的轨迹图 ----
    mf_configs = [
        ("runs_mf0.8", "Mean Flow 0.8", '#27AE60', (2, 0), (unified_xlim, unified_ylim)),  # 使用统一范围
        ("runs_mf0.5", "Mean Flow 0.5", '#27AE60', (2, 1), (unified_xlim, unified_ylim)),  # 使用统一范围
    ]
    
    for idx, (key, name, color, pos, limits) in enumerate(mf_configs):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        if key in pred_trajs_dict:
            pred_trajs = pred_trajs_dict[key]
            plot_trajectory(ax, gt_traj, pred_trajs, goal_point, 
                          title=name,
                          show_legend=False,  # 不再在子图中显示图例
                          pred_color=color,
                          pred_label=name,
                          set_limits=limits)  # 传入坐标轴范围
        else:
            ax.text(0.5, 0.5, f"No data for {name}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            # 移除标题
            # ax.set_title(name, fontsize=18, fontweight='bold')
            ax.axis("off")

    # ---- 在右下角绘制统一图例 ----
    ax_legend = fig.add_subplot(gs[2, 2])
    ax_legend.axis("off")
    
    # 创建图例元素
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=3, label='GT Trajectory'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=12, label='Start', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
               markersize=12, label='GT End', linestyle='None'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='orange', 
               markeredgecolor='black', markeredgewidth=0.5, markersize=18, 
               label='Goal', linestyle='None'),
        Line2D([0], [0], color='#2980B9', linewidth=2.5, alpha=0.8, label='Flow Matching'),
        Line2D([0], [0], color='#27AE60', linewidth=2.5, alpha=0.8, label='Mean Flow'),
    ]
    
    # 绘制图例
    legend = ax_legend.legend(handles=legend_elements, loc='center', 
                             fontsize=16, framealpha=0.95, 
                             fancybox=True, shadow=True,
                             edgecolor='#BDC3C7', facecolor='white')
    legend.get_frame().set_linewidth(1.5)

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


def load_model_from_checkpoint(ckpt_path, device, memory_size=8, predict_size=24):
    """
    智能加载模型，自动检测checkpoint类型（Flow或MeanFlow）
    
    Returns:
        model: 加载好的模型
        model_type: 'flow' 或 'meanflow'
    """
    # 先加载checkpoint查看参数
    ckpt = torch.load(ckpt_path, map_location=device)
    new_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    
    # 检测模型类型：通过cond_pos_embed的大小来判断
    if 'cond_pos_embed.position_embedding.weight' in new_ckpt:
        cond_pos_size = new_ckpt['cond_pos_embed.position_embedding.weight'].shape[0]
        # Flow模型: memory_size * 16 + 4 = 8 * 16 + 4 = 132
        # MeanFlow模型: memory_size * 16 + 6 = 8 * 16 + 6 = 134
        if cond_pos_size == memory_size * 16 + 6:
            model_type = 'meanflow'
        elif cond_pos_size == memory_size * 16 + 4:
            model_type = 'flow'
        else:
            # 尝试根据其他特征判断，默认使用Flow
            print(f"警告: 无法确定模型类型 (cond_pos_size={cond_pos_size})，默认使用Flow模型")
            model_type = 'flow'
    else:
        # 如果没有这个参数，默认使用Flow
        model_type = 'flow'
    
    # 根据类型创建模型
    if model_type == 'meanflow':
        # 根据路径判断flow_ratio
        if 'mf0.8' in ckpt_path or 'runs_mf0.8' in ckpt_path:
            flow_ratio = 0.8
        elif 'mf0.5' in ckpt_path or 'runs_mf0.5' in ckpt_path:
            flow_ratio = 0.5
        else:
            flow_ratio = 0.5  # 默认值
        
        model = NavDP_Policy_MeanFlow(
            image_size=224,
            memory_size=memory_size,
            predict_size=predict_size,
            temporal_depth=8,
            heads=8,
            token_dim=384,
            device=device,
            solver_steps=1,
            flow_ratio=flow_ratio,
        ).to(device)
    else:
        model = NavDP_Policy_Flow(
            image_size=224,
            memory_size=memory_size,
            predict_size=predict_size,
            temporal_depth=8,
            heads=8,
            token_dim=384,
            device=device,
            solver_steps=1,
        ).to(device)
    
    # 加载checkpoint，只加载匹配的参数
    model_dict = model.state_dict()
    matched_dict = {}
    skipped_keys = []
    
    for k, v in new_ckpt.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                matched_dict[k] = v
            else:
                skipped_keys.append(f"{k} (shape mismatch: {model_dict[k].shape} vs {v.shape})")
        else:
            skipped_keys.append(f"{k} (not in model)")
    
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict, strict=False)
    
    if skipped_keys:
        print(f"  跳过 {len(skipped_keys)} 个不匹配的参数")
    
    model.eval()
    return model, model_type


def main():
    # ---------- config ----------
    # 三个checkpoint路径（使用最新的checkpoint）
    ckpt_paths = {
        "runs_fm": "/mnt/houjunyi/MeanFlow_traj/runs_fm/navdpflow_step_149000.pt",
        "runs_mf0.8": "/mnt/houjunyi/MeanFlow_traj/runs_mf0.8/checkpoints/navdpflow_step_50000.pt",
        "runs_mf0.5": "/mnt/houjunyi/MeanFlow_traj/runs_mf0.5/checkpoints/navdpflow_step_50000.pt",
    }
    save_root = "tests/results_checkpoint_comparison"
    os.makedirs(save_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---------- dataset ----------
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

    # ---------- 加载三个模型 ----------
    models = {}
    for checkpoint_key, ckpt_path in ckpt_paths.items():
        if not os.path.exists(ckpt_path):
            print(f"警告: checkpoint不存在，跳过: {ckpt_path}")
            continue
        
        print(f"\n加载checkpoint: {checkpoint_key} from {ckpt_path}")
        try:
            model, model_type = load_model_from_checkpoint(
                ckpt_path, 
                device, 
                memory_size=8, 
                predict_size=24
            )
            models[checkpoint_key] = model
            print(f"✓ 成功加载 {checkpoint_key} (类型: {model_type})")
        except Exception as e:
            print(f"✗ 加载失败 ({checkpoint_key}): {e}")
            continue

    if not models:
        print("错误: 没有成功加载任何模型")
        return

    # ---------- inference ----------
    for step, batch in enumerate(tqdm(dataloader, desc="Testing")):
        traj_target = batch["batch_labels"].to(device)   # (1,T,3)
        input_images = batch["batch_rgb"].to(device)     # (1,mem,H,W,3)
        input_depths = batch["batch_depth"].to(device)   # (1,mem,H,W,1)
        goal_point = batch["batch_pg"].to(device)        # (1,3)

        # numpy 化（这些不依赖于checkpoint）
        gt_traj = traj_target[0].cpu().numpy()
        goal = goal_point[0].cpu().numpy()
        rgb = input_images[0, -1].cpu().numpy()
        depth = input_depths[0].cpu().numpy().squeeze()

        # 为每个checkpoint生成轨迹
        pred_trajs_dict = {}
        for checkpoint_key, model in models.items():
            if checkpoint_key == "runs_fm":
                # Flow Matching模型：生成1步、3步、10步的结果
                solver_steps_list = [1, 3, 10]
                original_steps = model.solver_steps
                
                for solver_steps in solver_steps_list:
                    model.solver_steps = solver_steps
                    with torch.no_grad():
                        trajs, _, _, _ = model.predict_pointgoal_action(
                            goal_point=goal_point,
                            input_images=input_images,
                            input_depths=input_depths,
                            sample_num=8,
                        )
                    pred_trajs_dict[f"runs_fm_{solver_steps}"] = trajs[0]
                
                # 恢复原始solver_steps
                model.solver_steps = original_steps
            else:
                # Mean Flow模型：单步生成
                with torch.no_grad():
                    trajs, _, _, _ = model.predict_pointgoal_action(
                        goal_point=goal_point,
                        input_images=input_images,
                        input_depths=input_depths,
                        sample_num=8,
                    )
                pred_trajs_dict[checkpoint_key] = trajs[0]

        # 保存结果图（包含RGB、Depth、Flow Matching不同步数和Mean Flow的轨迹图）
        save_results_figure(rgb, depth, gt_traj, pred_trajs_dict, goal, 
                          os.path.join(save_root, "results"), step)

        # 另外保存txt（使用Flow Matching 1步的结果）
        if "runs_fm_1" in pred_trajs_dict:
            save_traj_txt(gt_traj, pred_trajs_dict["runs_fm_1"], goal, 
                         os.path.join(save_root, "txt"), step)
        elif len(pred_trajs_dict) > 0:
            first_key = list(pred_trajs_dict.keys())[0]
            save_traj_txt(gt_traj, pred_trajs_dict[first_key], goal, 
                         os.path.join(save_root, "txt"), step)


if __name__ == "__main__":
    main()
