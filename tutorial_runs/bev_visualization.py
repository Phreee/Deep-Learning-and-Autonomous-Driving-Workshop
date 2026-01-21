"""
BEV (Bird's Eye View) Visualization Demo
演示透视图像到BEV俯视图的转换概念

目的：让学员理解BEV在智能驾驶中的作用，不涉及复杂的模型训练
方法：使用简化的几何投影（IPM - Inverse Perspective Mapping）
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os

def create_perspective_driving_scene():
    """
    创建一个简单的透视图驾驶场景（前视相机视角）
    包含：道路、车道线、车辆
    """
    # 创建640x480画布（前视相机典型分辨率）
    img = Image.new('RGB', (640, 480), color=(135, 206, 235))  # 天空蓝
    draw = ImageDraw.Draw(img)
    
    # 绘制道路（梯形透视效果）
    road_color = (80, 80, 80)  # 沥青灰
    # 道路边界：近处宽（底部），远处窄（消失点）
    road_points = [
        (150, 480),  # 左下
        (490, 480),  # 右下
        (360, 200),  # 右上（消失点附近）
        (280, 200),  # 左上（消失点附近）
    ]
    draw.polygon(road_points, fill=road_color)
    
    # 绘制车道线（透视效果：远小近大）
    lane_color = (255, 255, 255)
    
    # 中心虚线（透视收缩）
    for i in range(10):
        y_start = 480 - i * 30
        y_end = y_start - 20
        # 计算透视收缩的x位置
        center_x = 320 + (y_start - 480) * 0.0  # 收敛到中心
        width = max(2, 5 - i * 0.3)  # 远处更细
        draw.line([(center_x, y_start), (center_x, y_end)], 
                  fill=lane_color, width=int(width))
    
    # 左右边界线
    draw.line([(165, 480), (285, 200)], fill=lane_color, width=3)  # 左边界
    draw.line([(475, 480), (355, 200)], fill=lane_color, width=3)  # 右边界
    
    # 绘制前方车辆（远处，透视较小）
    car_points = [
        (290, 280), (350, 280),  # 车顶
        (360, 320), (280, 320)   # 车底
    ]
    draw.polygon(car_points, fill=(255, 0, 0))  # 红色车辆
    
    # 绘制近处车辆（左侧，透视较大）
    car2_points = [
        (180, 400), (240, 400),
        (250, 470), (170, 470)
    ]
    draw.polygon(car2_points, fill=(0, 0, 255))  # 蓝色车辆
    
    # 添加文字标注
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "Perspective View (Front Camera)", fill=(255, 255, 255), font=font)
    draw.text((10, 450), "Near: Wide", fill=(255, 255, 0), font=font)
    draw.text((10, 220), "Far: Narrow", fill=(255, 255, 0), font=font)
    
    return np.array(img)

def create_bev_driving_scene():
    """
    创建对应的BEV俯视图场景
    空间关系保持真实，无透视变形
    """
    # 创建640x480画布（BEV俯视视角）
    img = Image.new('RGB', (640, 480), color=(180, 180, 180))  # 灰色背景
    draw = ImageDraw.Draw(img)
    
    # 绘制道路（矩形，无透视变形）
    road_color = (80, 80, 80)
    road_rect = [200, 50, 440, 480]  # 恒定宽度
    draw.rectangle(road_rect, fill=road_color)
    
    # 绘制车道线（平行，无收缩）
    lane_color = (255, 255, 255)
    
    # 中心虚线（均匀分布）
    for i in range(15):
        y_start = 50 + i * 30
        y_end = y_start + 20
        draw.line([(320, y_start), (320, y_end)], fill=lane_color, width=3)
    
    # 左右边界线（平行）
    draw.line([(215, 50), (215, 480)], fill=lane_color, width=3)  # 左边界
    draw.line([(425, 50), (425, 480)], fill=lane_color, width=3)  # 右边界
    
    # 绘制前方车辆（真实大小，无透视缩小）
    car_rect = [290, 200, 350, 260]  # 车辆1（远处，但BEV中保持真实大小）
    draw.rectangle(car_rect, fill=(255, 0, 0))  # 红色车辆
    
    # 绘制近处车辆（左侧，真实大小）
    car2_rect = [220, 360, 280, 420]  # 车辆2
    draw.rectangle(car2_rect, fill=(0, 0, 255))  # 蓝色车辆
    
    # 自车位置（底部中心）
    ego_rect = [300, 440, 340, 480]
    draw.rectangle(ego_rect, fill=(0, 255, 0))  # 绿色自车
    
    # 绘制坐标网格（显示真实空间度量）
    grid_color = (150, 150, 150)
    for i in range(5, 48, 5):  # 每5米一条网格线
        y = 50 + i * 9
        draw.line([(200, y), (440, y)], fill=grid_color, width=1)
    
    # 添加文字标注
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    draw.text((10, 10), "BEV (Bird's Eye View)", fill=(0, 0, 0), font=font)
    draw.text((10, 450), "Ego Vehicle", fill=(0, 255, 0), font=font)
    draw.text((450, 100), "50m", fill=(0, 0, 0), font=small_font)
    draw.text((450, 200), "35m", fill=(0, 0, 0), font=small_font)
    draw.text((450, 300), "20m", fill=(0, 0, 0), font=small_font)
    draw.text((450, 400), "5m", fill=(0, 0, 0), font=small_font)
    
    return np.array(img)

def create_transformation_diagram():
    """
    创建透视图到BEV的转换过程示意图
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 透视图
    perspective = create_perspective_driving_scene()
    axes[0].imshow(perspective)
    axes[0].set_title('Input: Perspective View\n(Front Camera)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    axes[0].text(320, 520, 'Problem: Distance distortion, difficult to measure',
                ha='center', fontsize=11, color='red', 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 2. 转换过程（概念图）
    axes[1].text(0.5, 0.6, 'Transformation Process', ha='center', va='center',
                fontsize=16, fontweight='bold', transform=axes[1].transAxes)
    axes[1].text(0.5, 0.45, '↓', ha='center', va='center',
                fontsize=40, transform=axes[1].transAxes)
    
    transformation_text = """
    Methods:
    • Geometric Projection (IPM)
    • Depth Estimation + 3D
    • End-to-End Learning (LSS)
    • Transformer (BEVFormer)
    
    Key Challenge:
    • Multi-camera fusion
    • Depth ambiguity
    • Occlusion handling
    """
    axes[1].text(0.5, 0.2, transformation_text, ha='center', va='center',
                fontsize=10, family='monospace', transform=axes[1].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1].axis('off')
    
    # 3. BEV结果
    bev = create_bev_driving_scene()
    axes[2].imshow(bev)
    axes[2].set_title('Output: BEV Representation\n(Bird\'s Eye View)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    axes[2].text(320, 520, 'Benefit: Real distance, easy to plan',
                ha='center', fontsize=11, color='green', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_comparison_visualization():
    """
    创建详细的对比可视化
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)
    
    # 第一行：并排对比
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    perspective = create_perspective_driving_scene()
    bev = create_bev_driving_scene()
    
    ax1.imshow(perspective)
    ax1.set_title('Perspective View (Camera)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(bev)
    ax2.set_title('BEV View (Top-Down)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 第二行：特性对比表格
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    comparison_data = [
        ['Feature', 'Perspective View', 'BEV View'],
        ['Spatial Representation', '❌ Near big, far small', '✅ Real metric space'],
        ['Distance Measurement', '❌ Hard to measure', '✅ Direct measurement'],
        ['Multi-Camera Fusion', '❌ Different coordinate', '✅ Unified coordinate'],
        ['Planning Friendly', '❌ Need projection', '✅ Direct mapping'],
        ['Typical Use Case', 'End-to-End learning', 'Structured perception']
    ]
    
    table = ax3.table(cellText=comparison_data, cellLoc='left',
                     colWidths=[0.25, 0.375, 0.375],
                     loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 设置表头样式
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 第三行：真实系统示例
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    real_systems = """
    Real-World BEV Systems:
    
    Tesla FSD (2020+):  Pure Vision BEV → 8 cameras → Unified BEV feature → Occupancy network
    
    XPeng NGP (2021+):  Vision + LiDAR → BEV fusion → Trajectory prediction
    
    Baidu Apollo (2022+):  Multi-sensor BEV → Occupancy prediction → Planning
    
    Technical Evolution:  2016-2019: End-to-End → 2020+: BEV-based Structured Perception
    """
    
    ax4.text(0.5, 0.5, real_systems, ha='center', va='center',
            fontsize=11, family='monospace', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    return fig

def main():
    """
    主函数：生成所有BEV可视化
    """
    print("=" * 60)
    print("BEV Visualization Demo")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[1/3] Generating transformation diagram...")
    fig1 = create_transformation_diagram()
    output_path1 = os.path.join(output_dir, 'bev_transform_demo.png')
    fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path1}")
    
    print("\n[2/3] Generating detailed comparison...")
    fig2 = create_comparison_visualization()
    output_path2 = os.path.join(output_dir, 'bev_comparison.png')
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path2}")
    
    print("\n[3/3] Generating summary report...")
    summary = f"""
BEV Visualization Summary
{'=' * 60}

Generated Files:
• {output_path1}
• {output_path2}

Key Concepts Demonstrated:
1. Perspective View: Camera's natural view with distance distortion
2. BEV View: Top-down view with real metric space
3. Transformation: Multiple methods (IPM, Learning-based)

Educational Value:
• Understand why BEV is important in autonomous driving
• See the difference between perspective and BEV
• Learn the technical evolution (End-to-End → BEV)

Next Steps:
• Review Chapter 3.4 for technical details
• Compare with Chapter 4.2 (End-to-End approach)
• Understand the trade-offs between different approaches

Note: This is a conceptual demo using simplified geometry.
Real BEV systems use deep learning (LSS, BEVFormer, etc.)
and require multi-camera calibration and training.
"""
    print(summary)
    
    summary_path = os.path.join(output_dir, 'bev_visualization_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"✅ Saved: {summary_path}")
    
    print("\n" + "=" * 60)
    print("✅ BEV Visualization completed successfully!")
    print("=" * 60)
    print("\nPlease review the generated images to understand:")
    print("  • How perspective view differs from BEV")
    print("  • Why BEV is important for planning")
    print("  • How real systems use BEV")
    
    plt.close('all')

if __name__ == "__main__":
    main()
