"""
创建轨迹预测动态演示 GIF
展示 BEV 视角下的轨迹预测过程
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import torch
from nuscenes_trajectory_train import LSTMTrajectoryPredictor, generate_synthetic_trajectories, TrajectoryDataset


def create_trajectory_animation(model_path, output_path, num_samples=3, fps=10):
    """
    创建轨迹预测动画 GIF
    
    Args:
        model_path: 训练好的模型路径
        output_path: 输出 GIF 路径
        num_samples: 展示的样本数量
        fps: 帧率
    """
    # 加载模型
    device = torch.device("cpu")
    model = LSTMTrajectoryPredictor(input_size=5, hidden_size=64, num_layers=2, output_size=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # 生成测试数据
    print("生成测试轨迹数据...")
    trajectories = generate_synthetic_trajectories(num_samples=500, history_frames=20, future_frames=30)
    dataset = TrajectoryDataset(trajectories[-num_samples:])  # 取最后几个样本
    
    # 准备动画数据
    samples_data = []
    with torch.no_grad():
        for i in range(num_samples):
            history, future = dataset[i]
            history_input = history.unsqueeze(0).to(device)
            prediction = model(history_input, future_frames=30)
            
            samples_data.append({
                'history': history[:, :2].numpy(),  # [20, 2]
                'future_gt': future.numpy(),  # [30, 2]
                'future_pred': prediction[0].cpu().numpy(),  # [30, 2]
            })
    
    # 创建动画
    fig, axes = plt.subplots(1, num_samples, figsize=(6*num_samples, 6))
    if num_samples == 1:
        axes = [axes]
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 初始化绘图元素
    plot_elements = []
    for idx, (ax, data) in enumerate(zip(axes, samples_data)):
        # 计算误差
        ade = np.mean(np.linalg.norm(data['future_pred'] - data['future_gt'], axis=1))
        fde = np.linalg.norm(data['future_pred'][-1] - data['future_gt'][-1])
        
        # 自车位置（原点）
        ego_circle = Circle((0, 0), 0.5, color='red', alpha=0.8, zorder=5)
        ax.add_patch(ego_circle)
        
        # 历史轨迹（固定显示）
        hist_line, = ax.plot(data['history'][:, 0], data['history'][:, 1], 
                             'g-', linewidth=3, label='历史轨迹', alpha=0.8)
        
        # 未来轨迹（动态显示）
        gt_line, = ax.plot([], [], 'b-', linewidth=3, label='真实未来', alpha=0.8)
        pred_line, = ax.plot([], [], 'r--', linewidth=3, label='预测未来', alpha=0.8)
        
        # 当前位置点
        gt_point, = ax.plot([], [], 'bo', markersize=10, zorder=6)
        pred_point, = ax.plot([], [], 'ro', markersize=10, zorder=6)
        
        # 设置坐标轴
        all_x = np.concatenate([data['history'][:, 0], data['future_gt'][:, 0]])
        all_y = np.concatenate([data['history'][:, 1], data['future_gt'][:, 1]])
        margin = 5
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
        
        ax.set_xlabel('X (米)', fontsize=12)
        ax.set_ylabel('Y (米)', fontsize=12)
        ax.set_title(f'样本{idx+1}: ADE={ade:.2f}m, FDE={fde:.2f}m', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plot_elements.append({
            'gt_line': gt_line,
            'pred_line': pred_line,
            'gt_point': gt_point,
            'pred_point': pred_point,
            'data': data
        })
    
    # 动画更新函数
    def update(frame):
        """更新每一帧"""
        for elem in plot_elements:
            if frame < 30:  # 显示未来30帧
                # 更新轨迹线
                elem['gt_line'].set_data(elem['data']['future_gt'][:frame+1, 0], 
                                        elem['data']['future_gt'][:frame+1, 1])
                elem['pred_line'].set_data(elem['data']['future_pred'][:frame+1, 0], 
                                          elem['data']['future_pred'][:frame+1, 1])
                
                # 更新当前点
                elem['gt_point'].set_data([elem['data']['future_gt'][frame, 0]], 
                                         [elem['data']['future_gt'][frame, 1]])
                elem['pred_point'].set_data([elem['data']['future_pred'][frame, 0]], 
                                           [elem['data']['future_pred'][frame, 1]])
            else:  # 暂停几帧以便观察
                pass
        
        return [e['gt_line'] for e in plot_elements] + \
               [e['pred_line'] for e in plot_elements] + \
               [e['gt_point'] for e in plot_elements] + \
               [e['pred_point'] for e in plot_elements]
    
    # 创建动画（30帧预测 + 10帧暂停）
    print("创建动画...")
    anim = animation.FuncAnimation(fig, update, frames=40, interval=100, blit=True, repeat=True)
    
    # 保存为 GIF
    print(f"保存 GIF 到 {output_path}...")
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=100)
    plt.close()
    
    print(f"✓ 轨迹预测动画已保存至: {output_path}")
    print(f"  - 包含 {num_samples} 个样本")
    print(f"  - 总帧数: 40 帧（30帧预测 + 10帧暂停）")
    print(f"  - 帧率: {fps} fps")
    print(f"  - 时长: {40/fps:.1f} 秒（循环播放）")


def main():
    """主函数"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "tutorial_runs", "output", "nuscenes_trajectory_model.pth")
    output_path = os.path.join(base_dir, "tutorial_runs", "output", "trajectory_prediction_demo.gif")
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行 nuscenes_trajectory_train.py 训练模型")
        return
    
    print("=" * 60)
    print("创建轨迹预测动态演示 GIF")
    print("=" * 60)
    
    # 创建动画（3个样本）
    create_trajectory_animation(
        model_path=model_path,
        output_path=output_path,
        num_samples=3,
        fps=10
    )
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
