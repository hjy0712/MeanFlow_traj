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

def plot_loss_curve(steps, values, save_path="loss_curve.png", title="Training Loss Curve", max_step=10000):
    """
    绘制美观的 loss 曲线图
    
    Args:
        steps: 训练步数
        values: loss 值
        save_path: 保存路径
        title: 图表标题
        max_step: 最大步数，只绘制0到max_step的数据，None表示绘制全部
    """
    # 过滤数据，只保留指定步数范围内的数据
    if max_step is not None:
        mask = (steps >= 0) & (steps <= max_step)
        steps = steps[mask]
        values = values[mask]
        print(f"已过滤数据，只显示 0 到 {max_step} 步，共 {len(steps)} 个数据点")
    
    # 创建图形，使用更现代的样式
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('#FAFAFA')
    
    # 定义颜色方案（浅棕色和深棕色）
    raw_color = '#D2B48C'  # 浅棕色（Tan）
    smoothed_color = '#8B4513'  # 深棕色（Saddle Brown）
    
    # 计算平滑曲线
    smoothed = None
    smoothed_steps = None
    if len(values) > 10:
        window_size = max(10, len(values) // 50)
        smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
        smoothed_steps = steps[:len(smoothed)]
    
    # 先绘制原始数据曲线（带透明度）
    ax.plot(steps, values, 
            linewidth=4.0, 
            alpha=0.7,  # 调整真实数据曲线的透明度
            color=raw_color,
            label='Raw Training Loss',
            zorder=2)
    
    # 绘制平滑曲线（不透明，深棕色）
    if smoothed is not None:
        ax.plot(smoothed_steps, smoothed,
                linewidth=1.5,
                alpha=1.0,  # 完全不透明
                color=smoothed_color,
                label=f'Smoothed Loss (window={window_size})',
                zorder=4)
    
    # 标记最小loss点
    min_idx = np.argmin(values)
    min_loss = values[min_idx]
    min_step = steps[min_idx]
    ax.scatter(min_step, min_loss,
              s=200,
              color='#F39C12',
              marker='*',
              edgecolors='white',
              linewidths=2,
              zorder=5,
              label=f'Minimum Loss: {min_loss:.6f}')
    
    # 添加最小点的标注线
    ax.axvline(min_step, color='#F39C12', linestyle='--', 
              linewidth=1.5, alpha=0.5, zorder=1)
    ax.axhline(min_loss, color='#F39C12', linestyle='--', 
              linewidth=1.5, alpha=0.5, zorder=1)
    
    # 设置标题和标签（更现代的设计）
    # ax.set_title(title, fontsize=22, fontweight='bold', pad=25, color='#2C3E50')
    ax.set_xlabel('Training Step', fontsize=22, fontweight='bold', color='#34495E', labelpad=10)
    ax.set_ylabel('Loss', fontsize=22, fontweight='bold', color='#34495E', labelpad=10)
    
    # 设置网格（更精细的样式）
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    # 设置坐标轴颜色和字体大小（增大数字字体）
    ax.tick_params(colors='#34495E', labelsize=22)
    
    # 设置图例（更美观的样式，增大字体）
    legend = ax.legend(loc='upper right', fontsize=22, 
                      framealpha=0.95, 
                      shadow=True,
                      fancybox=True,
                      edgecolor='#BDC3C7',
                      facecolor='white')
    legend.get_frame().set_linewidth(1.5)
    
    # 美化坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # 添加统计信息文本框（放在右边，图例下方）
    final_loss = values[-1]
    final_step = steps[-1]
    mean_loss = np.mean(values)
    std_loss = np.std(values)
    
    stats_text = f'Statistics\n'
    stats_text += f'━━━━━━━━━━━━━━━━━━━━\n'
    stats_text += f'Min Loss:  {min_loss:.6f}\n'
    stats_text += f'           (Step {min_step})\n'
    stats_text += f'━━━━━━━━━━━━━━━━━━━━\n'
    stats_text += f'Final Loss: {final_loss:.6f}\n'
    stats_text += f'            (Step {final_step})\n'
    stats_text += f'━━━━━━━━━━━━━━━━━━━━\n'
    stats_text += f'Mean: {mean_loss:.6f}\n'
    stats_text += f'Std:  {std_loss:.6f}\n'
    stats_text += f'━━━━━━━━━━━━━━━━━━━━\n'
    stats_text += f'Total Steps: {final_step}'
    
    # 将统计信息框放在右边（x=0.98），图例下方（y=0.65）
    ax.text(0.98, 0.65, stats_text,
            transform=ax.transAxes,
            fontsize=20,  # 增大统计信息字体
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.8', 
                     facecolor='white', 
                     edgecolor='#BDC3C7',
                     linewidth=1.5,
                     alpha=0.95),
            family='monospace',
            color='#2C3E50')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片（高质量）
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
               facecolor='#FAFAFA', edgecolor='none')
    print(f"\n图表已保存到: {save_path}")
    
    # 显示图片
    plt.show()

def main():
    # 事件文件路径
    event_file = "/mnt/houjunyi/MeanFlow_traj/runs_fm/tensorboard_logs/events.out.tfevents.1759403084.uniubi-bj-gpu004.2924905.0"
    
    if not os.path.exists(event_file):
        print(f"错误: 文件不存在: {event_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("TensorBoard Loss 曲线绘制工具")
    print("=" * 60)
    
    # 加载数据
    try:
        steps, values = load_tensorboard_data(event_file, scalar_name="Loss/train")
    except Exception as e:
        print(f"加载数据时出错: {e}")
        sys.exit(1)
    
    # 绘制曲线（只绘制0到10000步）
    output_path = "/mnt/houjunyi/MeanFlow_traj/runs_fm/loss_curve.png"
    plot_loss_curve(steps, values, 
                   save_path=output_path,
                #    title="MeanFlow Training Loss Curve (Steps 0-10000)",
                   max_step=10000)
    
    print("\n完成!")

if __name__ == "__main__":
    main()

