"""
测试文件夹中真实rgb的轨迹预测
加载两个模型：FlowMatching模型 + DDPM模型
分别推理并可视化轨迹
"""
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from depth_anything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from models_traj.policy_network import NavDP_Policy_Flow
from navdp_test.policy_network import NavDP_Policy


DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


# ===================== 图像处理函数 =====================
def process_image(images, image_size=224):
    assert len(images.shape) == 4
    H, W, C = images.shape[1], images.shape[2], images.shape[3]
    prop = image_size / max(H, W)
    return_images = []
    for img in images:
        resize_image = cv2.resize(img, (-1, -1), fx=prop, fy=prop)
        pad_width = max((image_size - resize_image.shape[1]) // 2, 0)
        pad_height = max((image_size - resize_image.shape[0]) // 2, 0)
        pad_image = np.pad(resize_image,
                           ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
                           mode='constant', constant_values=0)
        resize_image = cv2.resize(pad_image, (image_size, image_size))
        resize_image = np.array(resize_image).astype(np.float32) / 255.0
        return_images.append(resize_image)
    return np.array(return_images)


def process_depth(depths, image_size=224):
    assert len(depths.shape) == 4
    depths[depths == np.inf] = 0
    H, W, C = depths.shape[1], depths.shape[2], depths.shape[3]
    prop = image_size / max(H, W)
    return_depths = []
    for depth in depths:
        resize_depth = cv2.resize(depth, (-1, -1), fx=prop, fy=prop)
        pad_width = max((image_size - resize_depth.shape[1]) // 2, 0)
        pad_height = max((image_size - resize_depth.shape[0]) // 2, 0)
        pad_depth = np.pad(resize_depth, ((pad_height, pad_height), (pad_width, pad_width)),
                           mode='constant', constant_values=0)
        resize_depth = cv2.resize(pad_depth, (image_size, image_size))
        resize_depth[resize_depth > 5.0] = 0
        resize_depth[resize_depth < 0.1] = 0
        return_depths.append(resize_depth[:, :, np.newaxis])
    return np.array(return_depths)


def infer_depth(image_path):
    """推理深度图"""
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitb'
    dataset = 'hypersim'
    max_depth = 20
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(
        '/mnt/houjunyi/MeanFlow_traj/depth_anything/checkpoints/depth_anything_v2_metric_hypersim_vitb.pth',
        map_location='cpu'))
    model = model.to(DEVICE).eval()

    raw_img = cv2.imread(image_path)
    depth = model.infer_image(raw_img)
    return depth


# ===================== 可视化与保存 =====================
def save_results_figure(rgb, depth, pred_trajs, goal_point, save_dir, step, model_name):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    # Depth
    im = axes[1].imshow(depth, cmap="gray")
    axes[1].set_title("Depth")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Trajectory
    ax = axes[2]
    for i, traj in enumerate(pred_trajs):
        full_pred = np.vstack([[0, 0, 0], traj])
        ax.plot(full_pred[:, 0], full_pred[:, 1], 'k-', alpha=0.6, label="Pred Traj" if i == 0 else "")
        ax.scatter(full_pred[-1, 0], full_pred[-1, 1], color="black", s=50, marker="^")
    ax.scatter(goal_point[0], goal_point[1], color="blue", s=150, marker="*", label="Goal")

    ax.set_title(f"{model_name} Predicted Trajectories")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, f"{model_name}_result_step{step}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path}")


# ===================== 主函数 =====================
def main():
    # ---- Config ----
    old_ckpt = "/mnt/houjunyi/MeanFlow_traj/runs_fm/checkpoints/navdpflow_step_149000.pt"
    new_ckpt = "/mnt/houjunyi/MeanFlow_traj/navdp_test/navdp-cross-modal.ckpt"

    image_paths = [
        "/mnt/houjunyi/MeanFlow_traj/tests/home_pic/test.jpeg"
    ]
    save_root = "tests/results_compare"
    os.makedirs(save_root, exist_ok=True)

    goal_point = np.array([2.5, 0.0, 0.0])

    # ---- Load Model 1: FlowMatching ----
    print(f"Loading FlowMatching model from {old_ckpt}")
    model1 = NavDP_Policy_Flow(
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=8,
        heads=8,
        token_dim=384,
        device=DEVICE,
        solver_steps=5,
    ).to(DEVICE)
    ckpt1 = torch.load(old_ckpt, map_location=DEVICE)
    ckpt1 = {k.replace("module.", ""): v for k, v in ckpt1.items()}
    model1.load_state_dict(ckpt1, strict=False)
    model1.eval()

    # ---- Load Model 2: DDPM ----
    print(f"Loading DDPM model from {new_ckpt}")
    model2 = NavDP_Policy(
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=8,
        heads=8,
        token_dim=384,
        channels=3,
        device=DEVICE,
    ).to(DEVICE)
    ckpt2 = torch.load(new_ckpt, map_location=DEVICE)
    ckpt2 = {k.replace("module.", ""): v for k, v in ckpt2.items()}
    model2.load_state_dict(ckpt2, strict=False)
    model2.eval()

    # ---- Loop over images ----
    for step, img_path in enumerate(tqdm(image_paths, desc="Testing")):
        rgb_raw = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # 模拟 memory_size=8 帧
        rgb_seq = np.stack([rgb_raw for _ in range(8)], axis=0)
        rgb_input = process_image(rgb_seq, image_size=224)
        rgb_input_tensor = torch.tensor(rgb_input).unsqueeze(0).to(DEVICE)

        # 深度推理
        depth_map = infer_depth(img_path)
        depth_seq = np.stack([depth_map for _ in range(1)], axis=0)[..., np.newaxis]
        depth_input = process_depth(depth_seq, image_size=224)
        depth_input_tensor = torch.tensor(depth_input).to(DEVICE)

        goal_tensor = torch.tensor(goal_point).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # ---- 模型1推理 ----
            trajs1, _, _, _ = model1.predict_pointgoal_action(
                goal_point=goal_tensor,
                input_images=rgb_input_tensor,
                input_depths=depth_input_tensor,
                sample_num=8,
            )

            # ---- 模型2推理 ----
            trajs2, _, _, _ = model2.predict_pointgoal_action(
                goal_point=goal_tensor,
                input_images=rgb_input_tensor,
                input_depths=depth_input_tensor,
                sample_num=8,
            )

        pred_trajs1 = trajs1[0]
        pred_trajs2 = trajs2[0]

        save_results_figure(rgb_input[-1], depth_input[-1].squeeze(), pred_trajs1, goal_point, save_root, step, "MeanFlow")
        save_results_figure(rgb_input[-1], depth_input[-1].squeeze(), pred_trajs2, goal_point, save_root, step, "CrossModal")


if __name__ == "__main__":
    main()
