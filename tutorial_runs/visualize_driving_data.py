"""
行为克隆数据可视化工具
生成数据预览图和视频，展示摄像头采集的驾驶场景
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "tutorial_runs", "behavioral_cloning_data")
output_dir = os.path.join(base_dir, "tutorial_runs", "output")
os.makedirs(output_dir, exist_ok=True)

# 读取驾驶日志
log_file = os.path.join(data_dir, "driving_log_local.csv")
df = pd.read_csv(log_file)

# 修正列名（实际数据集只有 image 和 steering）
if 'image' in df.columns:
    df['center'] = df['image']
if 'steering' not in df.columns and 'angle' in df.columns:
    df['steering'] = df['angle']

print(f"总数据量: {len(df)} 条")
print(f"转向角度范围: [{df['steering'].min():.3f}, {df['steering'].max():.3f}]")

# 统计转向分布
straight = len(df[df['steering'].abs() < 0.1])
left = len(df[df['steering'] < -0.1])
right = len(df[df['steering'] > 0.1])
print(f"\n转向分布:")
print(f"  直行: {straight} ({straight/len(df)*100:.1f}%)")
print(f"  左转: {left} ({left/len(df)*100:.1f}%)")
print(f"  右转: {right} ({right/len(df)*100:.1f}%)")

# 1. 生成数据样本展示图（6张图像网格）
print("\n生成数据样本展示图...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('行为克隆数据样本预览 - 摄像头视角', fontsize=16, fontweight='bold')

# 选择不同场景的6张图像
sample_indices = [
    0,                    # 起点
    len(df)//6,          # 第1段
    len(df)//3,          # 第2段
    len(df)//2,          # 中点
    len(df)*2//3,        # 第4段
    len(df)-1            # 终点
]

for idx, ax in enumerate(axes.flat):
    sample_idx = sample_indices[idx]
    row = df.iloc[sample_idx]
    
    # 读取图像（处理路径）
    img_path = row['center'].strip()
    if not os.path.exists(img_path):
        # 尝试相对路径
        img_path = os.path.join(data_dir, 'IMG', os.path.basename(img_path))
    
    if os.path.exists(img_path):
        img = Image.open(img_path)
        ax.imshow(img)
        
        # 添加转向角度指示
        steering = row['steering']
        
        # 根据转向角度确定方向文本和颜色
        if abs(steering) < 0.1:
            direction = "直行"
            color = 'green'
        elif steering < 0:
            direction = "左转"
            color = 'blue'
        else:
            direction = "右转"
            color = 'red'
        
        # 添加信息文本框
        info_text = f"{direction}\n转向: {steering:.3f}"
        ax.text(0.02, 0.98, info_text, 
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                color='white',
                fontweight='bold')
        
        ax.set_title(f'样本 {sample_idx+1}/{len(df)}', fontsize=10)
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, f'图像未找到\n{os.path.basename(img_path)}', ha='center', va='center')
        ax.axis('off')

plt.tight_layout()
sample_path = os.path.join(output_dir, "driving_data_samples.png")
plt.savefig(sample_path, dpi=150, bbox_inches='tight')
print(f"已保存样本展示图: {sample_path}")

# 2. 生成转向角度分布图
print("\n生成转向角度分布图...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 直方图
axes[0].hist(df['steering'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='直行(0)')
axes[0].set_xlabel('转向角度', fontsize=12)
axes[0].set_ylabel('样本数量', fontsize=12)
axes[0].set_title('转向角度分布直方图', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 时序图（转向角度随帧变化）
axes[1].plot(range(len(df)), df['steering'], linewidth=0.5, alpha=0.7, color='steelblue')
axes[1].axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
axes[1].set_xlabel('帧序号', fontsize=12)
axes[1].set_ylabel('转向角度', fontsize=12)
axes[1].set_title('转向角度时序变化', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
dist_path = os.path.join(output_dir, "driving_data_distribution.png")
plt.savefig(dist_path, dpi=150, bbox_inches='tight')
print(f"已保存分布图: {dist_path}")

# 3. 生成连续帧序列图（模拟视频效果）
print("\n生成连续帧序列图...")
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('连续驾驶帧序列 - 模拟视频效果（每2帧取1帧）', fontsize=16, fontweight='bold')

start_idx = len(df) // 3  # 从1/3处开始，通常有更多转弯
frame_step = 2  # 每2帧取1帧

for idx, ax in enumerate(axes.flat):
    sample_idx = start_idx + idx * frame_step
    if sample_idx >= len(df):
        break
        
    row = df.iloc[sample_idx]
    img_path = row['center'].strip()
    if not os.path.exists(img_path):
        img_path = os.path.join(data_dir, 'IMG', os.path.basename(img_path))
    
    if os.path.exists(img_path):
        img = Image.open(img_path)
        ax.imshow(img)
        
        steering = row['steering']
        
        # 绘制转向指示箭头
        if abs(steering) > 0.05:
            arrow_length = abs(steering) * 0.3
            if steering < 0:  # 左转
                ax.arrow(0.3, 0.9, -arrow_length, 0,
                        transform=ax.transAxes,
                        head_width=0.1, head_length=0.05,
                        fc='blue', ec='blue', linewidth=2)
            else:  # 右转
                ax.arrow(0.7, 0.9, arrow_length, 0,
                        transform=ax.transAxes,
                        head_width=0.1, head_length=0.05,
                        fc='red', ec='red', linewidth=2)
        
        # 添加帧信息
        ax.text(0.02, 0.02, f'帧 {sample_idx}\n{steering:.3f}', 
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                color='black')
        
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
sequence_path = os.path.join(output_dir, "driving_sequence_frames.png")
plt.savefig(sequence_path, dpi=150, bbox_inches='tight')
print(f"已保存序列帧图: {sequence_path}")

# 4. 生成转向模式分析图
print("\n生成转向模式分析图...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('驾驶转向模式分析', fontsize=16, fontweight='bold')

# 只显示前500帧便于观察
display_length = min(500, len(df))
frames = range(display_length)

# 转向角度时间序列
axes[0].plot(frames, df['steering'][:display_length], 'b-', linewidth=1, alpha=0.7)
axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[0].fill_between(frames, 0, df['steering'][:display_length], 
                      where=(df['steering'][:display_length]>0), 
                      color='red', alpha=0.3, label='右转')
axes[0].fill_between(frames, 0, df['steering'][:display_length], 
                      where=(df['steering'][:display_length]<0), 
                      color='blue', alpha=0.3, label='左转')
axes[0].set_ylabel('转向角度', fontsize=12)
axes[0].set_title('转向角度时序变化（前500帧）', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 转向角度绝对值（转向强度）
axes[1].plot(frames, df['steering'][:display_length].abs(), 'purple', linewidth=1.5)
axes[1].fill_between(frames, 0, df['steering'][:display_length].abs(), 
                      alpha=0.3, color='purple')
axes[1].set_xlabel('帧序号', fontsize=12)
axes[1].set_ylabel('转向强度（绝对值）', fontsize=12)
axes[1].set_title('转向强度变化', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
timeline_path = os.path.join(output_dir, "driving_timeline.png")
plt.savefig(timeline_path, dpi=150, bbox_inches='tight')
print(f"已保存转向分析图: {timeline_path}")

print("\n" + "="*60)
print("数据可视化完成！")
print("="*60)
print(f"\n生成的文件:")
print(f"1. 数据样本展示: {os.path.basename(sample_path)}")
print(f"2. 转向分布图: {os.path.basename(dist_path)}")
print(f"3. 连续帧序列: {os.path.basename(sequence_path)}")
print(f"4. 时间序列分析: {os.path.basename(timeline_path)}")
print("\n这些图像可以直接插入到教程文档中，帮助学员理解数据特征。")
