#!/usr/bin/env python3
"""
从 TensorBoard 事件文件中提取 loss 数据并绘制美观的曲线图
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import font_manager
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 设置美观的样式
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.style.use('seaborn-v0_8-darkgrid')

def load_tensorboard_data(event_file_path, scalar_name="Loss/train"):
    """
    从 TensorBoard 事件文件中加载标量数据
    
    Args:
        event_file_path: TensorBoard 事件文件路径
        scalar_name: 要提取的标量名称
    
    Returns:
        steps: 步数列表
        values: 对应的值列表
    """
    # 获取事件文件所在的目录
    log_dir = os.path.dirname(event_file_path)
    
    # 创建 EventAccumulator
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    # 获取所有可用的标量标签
    scalar_tags = ea.Tags()['scalars']
    print(f"可用的标量标签: {scalar_tags}")
    
    # 检查请求的标量是否存在
    if scalar_name not in scalar_tags:
        print(f"警告: '{scalar_name}' 不存在，可用的标签有: {scalar_tags}")
        if scalar_tags:
            scalar_name = scalar_tags[0]
            print(f"使用第一个可用标签: '{scalar_name}'")
        else:
            raise ValueError("没有找到任何标量数据")
    
    # 提取数据
    scalar_events = ea.Scalars(scalar_name)
    steps = [s.step for s in scalar_events]
    values = [s.value for s in scalar_events]
    
    print(f"成功加载 {len(steps)} 个数据点")
    print(f"步数范围: {min(steps)} - {max(steps)}")
    print(f"Loss 范围: {min(values):.6f} - {max(values):.6f}")
    
    return np.array(steps), np.array(values)

def plot_multiple_loss_curves(data_list, save_path="loss_curve.png", max_step=10000):
    """
    在同一张图中绘制多条 loss 曲线
    
    Args:
        data_list: 列表，每个元素是 (steps, values, label, color) 的元组
        save_path: 保存路径
        max_step: 最大步数，只绘制0到max_step的数据，None表示绘制全部
    """
    # 创建图形，使用更现代的样式
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('#FAFAFA')
    
    # 定义和谐的颜色方案（三种互补色）
    default_colors = [
        '#2980B9',  # 深蓝色
        '#27AE60',  # 深绿色
        '#D35400',  # 深橙色
    ]
    
    # 绘制每条曲线
    for idx, (steps, values, label, color) in enumerate(data_list):
        # 如果没有指定颜色，使用默认颜色
        if color is None:
            color = default_colors[idx % len(default_colors)]
        
        # 过滤数据，只保留指定步数范围内的数据
        if max_step is not None:
            mask = (steps >= 0) & (steps <= max_step)
            steps_filtered = steps[mask]
            values_filtered = values[mask]
        else:
            steps_filtered = steps
            values_filtered = values
        
        # 计算平滑曲线
        smoothed = None
        smoothed_steps = None
        if len(values_filtered) > 10:
            window_size = max(10, len(values_filtered) // 50)
            smoothed = np.convolve(values_filtered, np.ones(window_size)/window_size, mode='valid')
            smoothed_steps = steps_filtered[:len(smoothed)]
        
        # 如果有平滑曲线，绘制原始数据（透明）和平滑曲线（不透明）
        if smoothed is not None:
            # 绘制原始数据曲线（带透明度，不显示在图例中）
            ax.plot(steps_filtered, values_filtered, 
                    linewidth=3.0, 
                    alpha=0.3,  # 原始数据较透明
                    color=color,
                    zorder=2)
            # 绘制平滑曲线（不透明，更粗，显示在图例中）
            ax.plot(smoothed_steps, smoothed,
                    linewidth=2.5,
                    alpha=1.0,  # 平滑曲线完全不透明
                    color=color,
                    label=label,  # 只显示平滑曲线的标签
                    zorder=3)
        else:
            # 如果没有平滑曲线，只绘制原始数据并显示标签
            ax.plot(steps_filtered, values_filtered, 
                    linewidth=2.5,
                    alpha=1.0,
                    color=color,
                    label=label,
                    zorder=3)
    
    # 设置标题和标签（更现代的设计）
    ax.set_xlabel('Training Step', fontsize=22, fontweight='bold', color='#34495E', labelpad=10)
    ax.set_ylabel('Loss', fontsize=22, fontweight='bold', color='#34495E', labelpad=10)
    
    # 设置网格（更精细的样式）
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    # 设置坐标轴颜色和字体大小（增大数字字体）
    ax.tick_params(colors='#34495E', labelsize=22)
    
    # 设置图例（更美观的样式，增大字体）
    legend = ax.legend(loc='upper right', fontsize=18, 
                      framealpha=0.95, 
                      shadow=True,
                      fancybox=True,
                      edgecolor='#BDC3C7',
                      facecolor='white',
                      ncol=1)
    legend.get_frame().set_linewidth(1.5)
    
    # 美化坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片（高质量）
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
               facecolor='#FAFAFA', edgecolor='none')
    print(f"\n图表已保存到: {save_path}")
    
    # 显示图片
    plt.show()

def main():
    # 三个事件文件路径和对应的标签
    event_files = [
        ("/mnt/houjunyi/MeanFlow_traj/runs_fm/tensorboard_logs/events.out.tfevents.1759403084.uniubi-bj-gpu004.2924905.0", "Flow Matching", None),
        ("/mnt/houjunyi/MeanFlow_traj/runs_mf0.8/tensorboard_logs/events.out.tfevents.1760685546.uniubi-bj-gpu004.2819123.0", "Mean Flow 0.8", None),
        ("/mnt/houjunyi/MeanFlow_traj/runs_mf0.5/tensorboard_logs/events.out.tfevents.1760170162.uniubi-bj-gpu004.461982.0", "Mean Flow 0.5", None),
    ]
    
    print("=" * 60)
    print("TensorBoard Loss 曲线绘制工具 - 多曲线对比")
    print("=" * 60)
    
    # 加载所有数据
    data_list = []
    for event_file, label, color in event_files:
        if not os.path.exists(event_file):
            print(f"警告: 文件不存在，跳过: {event_file}")
            continue
        
        print(f"\n正在加载: {label}")
        try:
            steps, values = load_tensorboard_data(event_file, scalar_name="Loss/train")
            data_list.append((steps, values, label, color))
            print(f"✓ 成功加载 {label}")
        except Exception as e:
            print(f"✗ 加载数据时出错 ({label}): {e}")
            continue
    
    if not data_list:
        print("错误: 没有成功加载任何数据")
        sys.exit(1)
    
    # 绘制多条曲线（只绘制0到10000步）
    output_path = "/mnt/houjunyi/MeanFlow_traj/runs_fm/loss_curve_comparison.png"
    plot_multiple_loss_curves(data_list, 
                             save_path=output_path,
                             max_step=10000)
    
    print("\n完成!")

if __name__ == "__main__":
    main()

