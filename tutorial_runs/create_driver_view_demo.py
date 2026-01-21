"""
创建驾驶者主视角的轨迹预测演示 GIF
模拟车载摄像头视角，展示前方车辆及其预测轨迹
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon
import torch
from nuscenes_trajectory_train import LSTMTrajectoryPredictor, generate_synthetic_trajectories, TrajectoryDataset


def bev_to_camera_view(x, y, camera_height=1.5, focal_length=500):
    """
    将BEV坐标转换为相机视角坐标
    简化的透视投影
    """
    # 假设相机高度1.5米，俯仰角略微向下
    z = camera_height
    
    # 透视投影（简化版）
    if y > 0.1:  # 避免除零
        screen_x = (x / y) * focal_length + 320  # 屏幕中心640/2
        screen_y = (z / y) * focal_length + 100   # 地平线位置
        scale = focal_length / y  # 距离越远，物体越小
    else:
        screen_x = 320
        screen_y = 100
        scale = 10
    
    return screen_x, screen_y, scale


def draw_road(ax):
    """绘制道路背景"""
    # 天空
    sky = Rectangle((0, 0), 640, 200, facecolor='#87CEEB', edgecolor='none')
    ax.add_patch(sky)
    
    # 路面
    road = Rectangle((0, 200), 640, 280, facecolor='#555555', edgecolor='none')
    ax.add_patch(road)
    
    # 车道线（透视效果）
    lane_y_start = 200
    lane_y_end = 480
    
    # 中间虚线
    for i in range(5):
        y1 = lane_y_start + i * 60
        y2 = y1 + 30
        width_top = 10 - i * 1.5
        width_bottom = 15 - i * 2
        
        lane_dash = Polygon([
            [320 - width_top/2, y1],
            [320 + width_top/2, y1],
            [320 + width_bottom/2, y2],
            [320 - width_bottom/2, y2]
        ], facecolor='white', alpha=0.8, edgecolor='none')
        ax.add_patch(lane_dash)
    
    # 左车道线
    left_lane = Polygon([
        [100, lane_y_start],
        [110, lane_y_start],
        [50, lane_y_end],
        [40, lane_y_end]
    ], facecolor='white', edgecolor='none')
    ax.add_patch(left_lane)
    
    # 右车道线
    right_lane = Polygon([
        [530, lane_y_start],
        [540, lane_y_start],
        [600, lane_y_end],
        [590, lane_y_end]
    ], facecolor='white', edgecolor='none')
    ax.add_patch(right_lane)


def draw_vehicle(ax, x, y, color='blue', alpha=0.7, label_text=None):
    """在相机视角中绘制车辆"""
    screen_x, screen_y, scale = bev_to_camera_view(x, y)
    
    # 车辆尺寸随距离缩放
    width = scale * 2
    height = scale * 3
    
    # 绘制车辆矩形
    vehicle = Rectangle(
        (screen_x - width/2, screen_y - height/2),
        width, height,
        facecolor=color,
        edgecolor='white',
        linewidth=2,
        alpha=alpha
    )
    ax.add_patch(vehicle)
    
    # 添加标签
    if label_text:
        ax.text(screen_x, screen_y - height, label_text,
                ha='center', va='bottom', fontsize=8,
                color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    return vehicle


def draw_trajectory_projection(ax, trajectory, color='yellow', alpha=0.5):
    """在道路上绘制预测轨迹的投影"""
    screen_points = []
    for point in trajectory:
        x, y = point[0], point[1]
        if y > 0.5:  # 只显示前方的点
            screen_x, screen_y, scale = bev_to_camera_view(x, y)
            screen_points.append([screen_x, screen_y])
    
    if len(screen_points) > 1:
        screen_points = np.array(screen_points)
        ax.plot(screen_points[:, 0], screen_points[:, 1],
                color=color, linewidth=2, linestyle='--',
                alpha=alpha, marker='o', markersize=3)


def create_driver_view_animation(model_path, output_path, fps=10):
    """
    创建驾驶者视角的轨迹预测动画（多车场景，全部从数据生成）
    """
    # 加载模型
    device = torch.device("cpu")
    model = LSTMTrajectoryPredictor(input_size=5, hidden_size=64, num_layers=2, output_size=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # 生成多车场景数据
    print("生成多车测试场景...")
    np.random.seed(42)
    
    # 生成足够多的轨迹样本（包含不同运动模式）
    all_trajectories = generate_synthetic_trajectories(num_samples=200, history_frames=20, future_frames=30)
    
    # 准备车辆列表
    all_vehicles = []
    
    # 目标车辆（选择一个变道场景）- 进行预测
    # 选择第10个样本作为目标
    target_dataset = TrajectoryDataset([all_trajectories[10]])
    target_history, target_future = target_dataset[0]
    
    # 预测目标车辆轨迹
    with torch.no_grad():
        history_input = target_history.unsqueeze(0).to(device)
        target_prediction = model(history_input, future_frames=30)
        target_pred_traj = target_prediction[0].cpu().numpy()
    
    # 调整目标车辆位置到中心车道
    target_history_adj = target_history[:, :2].numpy()
    target_future_adj = target_future.numpy()
    target_pred_adj = target_pred_traj.copy()
    
    all_vehicles.append({
        'type': 'target',
        'history': target_history_adj,
        'future': target_future_adj,
        'prediction': target_pred_adj,
        'color_real': '#2E86DE',
        'color_pred': '#EE5A6F',
        'label': '目标'
    })
    
    # 左车道车辆（选择直行模式的样本，调整到左侧）
    for idx in [15, 25]:
        traj_data = TrajectoryDataset([all_trajectories[idx]])[0]
        history_pos = traj_data[0][:, :2].numpy()
        future_pos = traj_data[1].numpy()
        
        # 平移到左车道
        history_pos[:, 0] -= 2.5
        future_pos[:, 0] -= 2.5
        # 调整距离
        history_pos[:, 1] += 25
        future_pos[:, 1] += 25
        
        all_vehicles.append({
            'type': 'background',
            'history': history_pos,
            'future': future_pos,
            'color_real': '#4A90E2',
            'label': None
        })
    
    # 右车道车辆（调整到右侧）
    for idx in [35, 45]:
        traj_data = TrajectoryDataset([all_trajectories[idx]])[0]
        history_pos = traj_data[0][:, :2].numpy()
        future_pos = traj_data[1].numpy()
        
        # 平移到右车道
        history_pos[:, 0] += 2.5
        future_pos[:, 0] += 2.5
        # 调整距离
        history_pos[:, 1] += 20
        future_pos[:, 1] += 20
        
        all_vehicles.append({
            'type': 'background',
            'history': history_pos,
            'future': future_pos,
            'color_real': '#E2A04A',
            'label': None
        })
    
    # 前方同车道车辆（调整距离较远）
    traj_data = TrajectoryDataset([all_trajectories[55]])[0]
    history_pos = traj_data[0][:, :2].numpy()
    future_pos = traj_data[1].numpy()
    
    # 调整到前方较远位置
    history_pos[:, 1] += 40
    future_pos[:, 1] += 40
    
    all_vehicles.append({
        'type': 'background',
        'history': history_pos,
        'future': future_pos,
        'color_real': '#888888',
        'label': None
    })
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 计算目标车辆误差
    ade = np.mean(np.linalg.norm(target_pred_adj - target_future_adj, axis=1))
    fde = np.linalg.norm(target_pred_adj[-1] - target_future_adj[-1])
    
    # 初始化UI元素
    info_box = None
    
    def update(frame):
        """更新每一帧"""
        nonlocal info_box
        
        ax.clear()
        ax.set_xlim(0, 640)
        ax.set_ylim(0, 480)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 绘制道路背景
        draw_road(ax)
        
        # 调整时间映射：200帧映射到50帧的数据
        # 前60帧：历史阶段（20帧数据）
        # 60-180帧：预测阶段（30帧数据）
        # 180-200帧：暂停显示结果
        
        if frame < 60:
            # 历史阶段（60帧显示20帧数据）
            data_idx = int(frame * 20 / 60)
            data_idx = min(data_idx, 19)  # 确保不超出范围
            
            # 绘制所有车辆的历史位置
            for vehicle in all_vehicles:
                current_pos = vehicle['history'][data_idx]
                
                if vehicle['type'] == 'target':
                    draw_vehicle(ax, current_pos[0], current_pos[1],
                               color=vehicle['color_real'], label_text=vehicle['label'])
                else:
                    # 只绘制在合理视野范围内的车辆
                    if 5 < current_pos[1] < 70:
                        draw_vehicle(ax, current_pos[0], current_pos[1],
                                   color=vehicle['color_real'], alpha=0.5)
            
            status_text = f'历史轨迹回放 ({data_idx+1}/20)'
            
        elif frame < 180:
            # 预测阶段（120帧显示30帧数据）
            future_idx = int((frame - 60) * 30 / 120)
            future_idx = min(future_idx, 29)  # 确保不超出范围
            
            # 绘制所有车辆
            for vehicle in all_vehicles:
                if vehicle['type'] == 'target':
                    # 目标车辆显示预测
                    real_pos = vehicle['future'][future_idx]
                    pred_pos = vehicle['prediction'][future_idx]
                    
                    # 真实位置
                    draw_vehicle(ax, real_pos[0], real_pos[1],
                               color=vehicle['color_real'], alpha=0.9, label_text='真实')
                    # 预测位置
                    draw_vehicle(ax, pred_pos[0], pred_pos[1],
                               color=vehicle['color_pred'], alpha=0.7, label_text='预测')
                    
                    # 预测轨迹投影（前80%帧显示）
                    if frame < 144:  # 80% of 180
                        draw_trajectory_projection(ax, vehicle['prediction'][future_idx:],
                                                 color='#F9CA24', alpha=0.6)
                else:
                    # 背景车辆
                    current_pos = vehicle['future'][future_idx]
                    # 只绘制在合理视野范围内的车辆
                    if 5 < current_pos[1] < 70:
                        draw_vehicle(ax, current_pos[0], current_pos[1],
                                   color=vehicle['color_real'], alpha=0.5)
            
            # 计算当前误差
            real_pos = all_vehicles[0]['future'][future_idx]
            pred_pos = all_vehicles[0]['prediction'][future_idx]
            current_error = np.linalg.norm(pred_pos - real_pos)
            status_text = f'预测演示 ({future_idx+1}/30) | 当前误差: {current_error:.2f}m'
            
        else:
            # 暂停阶段（显示最终结果）
            # 显示最后一帧的状态
            for vehicle in all_vehicles:
                if vehicle['type'] == 'target':
                    real_pos = vehicle['future'][-1]
                    pred_pos = vehicle['prediction'][-1]
                    
                    draw_vehicle(ax, real_pos[0], real_pos[1],
                               color=vehicle['color_real'], alpha=0.9, label_text='真实')
                    draw_vehicle(ax, pred_pos[0], pred_pos[1],
                               color=vehicle['color_pred'], alpha=0.7, label_text='预测')
                else:
                    current_pos = vehicle['future'][-1]
                    if 5 < current_pos[1] < 70:
                        draw_vehicle(ax, current_pos[0], current_pos[1],
                                   color=vehicle['color_real'], alpha=0.5)
            
            status_text = '预测完成 - 最终结果展示'
        
        # 添加信息面板
        info_texts = [
            '🚗 驾驶者视角 - 多车场景轨迹预测（数据驱动）',
            status_text,
            f'平均位移误差(ADE): {ade:.2f}m',
            f'终点位移误差(FDE): {fde:.2f}m'
        ]
        
        for i, text in enumerate(info_texts):
            ax.text(10, 470 - i*20, text,
                   fontsize=10 if i == 0 else 9,
                   color='white',
                   weight='bold' if i == 0 else 'normal',
                   bbox=dict(boxstyle='round,pad=0.5',
                            facecolor='black', alpha=0.7))
        
        # 添加图例
        legend_y = 30
        legend_items = [
            ('蓝色车辆', '#2E86DE', '目标真实位置'),
            ('红色车辆', '#EE5A6F', '目标预测位置'),
            ('黄色虚线', '#F9CA24', '预测轨迹'),
            ('其他车辆', '#888888', '周围交通流（数据生成）')
        ]
        
        for i, (label, color, desc) in enumerate(legend_items):
            ax.add_patch(Rectangle((410, legend_y + i*25), 15, 10,
                                   facecolor=color, alpha=0.7))
            ax.text(430, legend_y + i*25 + 5, f'{desc}',
                   fontsize=7, color='white', va='center',
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='black', alpha=0.6))
        
        return []
    
    # 创建动画（60帧历史 + 120帧预测 + 20帧暂停 = 200帧，10fps = 20秒）
    print("创建动画...")
    total_frames = 200
    anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=100, blit=False, repeat=True)
    
    # 保存为 GIF
    print(f"保存 GIF 到 {output_path}...")
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=100)
    plt.close()
    
    print(f"✓ 驾驶者视角轨迹预测动画已保存")
    print(f"  - 视角: 车载摄像头主视角（多车场景）")
    print(f"  - 总帧数: {total_frames} 帧")
    print(f"  - 时长: {total_frames/fps:.1f} 秒（循环播放）")
    print(f"  - 场景: 6辆车（1辆目标车+5辆背景车），全部从数据生成")
    print(f"  - 运动模式: 变道、直行、转弯（基于合成轨迹数据）")
    print(f"  - 阶段分配: 6秒历史回放 + 12秒预测演示 + 2秒结果展示")


def main():
    """主函数"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "tutorial_runs", "output", "nuscenes_trajectory_model.pth")
    output_path = os.path.join(base_dir, "tutorial_runs", "output", "trajectory_driver_view_demo.gif")
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行 nuscenes_trajectory_train.py 训练模型")
        return
    
    print("=" * 60)
    print("创建驾驶者视角轨迹预测演示")
    print("=" * 60)
    
    create_driver_view_animation(
        model_path=model_path,
        output_path=output_path,
        fps=10
    )
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
