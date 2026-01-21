"""
BEV from Real Driving Data
从真实驾驶图像生成BEV俯视图

方法：单目深度估计 + IPM（逆透视映射）
数据：behavioral_cloning_data中的真实驾驶图像
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def simple_depth_estimation(image_rgb):
    """
    简化的深度估计（基于图像亮度和位置的启发式方法）
    
    注意：这是教学简化版本，真实系统使用深度学习模型（如MiDaS）
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # 创建深度图：基于垂直位置（越靠下越近）
    height, width = gray.shape
    depth_map = np.zeros_like(gray, dtype=np.float32)
    
    # 简单启发式：
    # 1. 垂直位置：底部近，顶部远
    # 2. 亮度：道路（暗）近，天空（亮）远
    for y in range(height):
        # 垂直位置权重：0（顶部，远）到 1（底部，近）
        vertical_weight = (height - y) / height
        
        # 结合亮度信息
        brightness_weight = 1.0 - (gray[y, :] / 255.0) * 0.3
        
        depth_map[y, :] = vertical_weight * brightness_weight
    
    # 归一化到 [0, 1]
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    
    return depth_map

def depth_to_bev(image_rgb, depth_map, bev_size=(400, 400)):
    """
    将透视图像和深度图转换为BEV俯视图
    
    参数：
        image_rgb: 原始RGB图像 (H, W, 3)
        depth_map: 深度图 (H, W)，值范围 [0, 1]，0=远，1=近
        bev_size: BEV输出尺寸 (height, width)
    
    返回：
        bev_image: BEV俯视图
        bev_depth: BEV深度图（用于可视化）
    """
    h, w, _ = image_rgb.shape
    bev_h, bev_w = bev_size
    
    # 初始化BEV图像和深度累积器
    bev_image = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    bev_depth = np.zeros((bev_h, bev_w), dtype=np.float32)
    bev_count = np.zeros((bev_h, bev_w), dtype=np.float32)
    
    # 相机参数（简化假设）
    focal_length = w * 1.2  # 焦距（像素单位）
    camera_height = 1.5  # 相机高度（米）
    pitch_angle = 0.0  # 俯仰角（弧度）
    
    # 遍历图像每个像素
    for v in range(h):
        for u in range(w):
            # 跳过天空区域（顶部1/3）
            if v < h * 0.3:
                continue
            
            # 获取深度值（转换为实际距离，米）
            depth_normalized = depth_map[v, u]
            if depth_normalized < 0.1:  # 跳过太远的点
                continue
            
            # 深度映射：将[0,1]映射到[50米, 5米]
            depth_meters = 50.0 - depth_normalized * 45.0
            
            # 图像坐标归一化到[-1, 1]
            u_norm = (u - w / 2) / (w / 2)
            v_norm = (v - h / 2) / (h / 2)
            
            # 简化的IPM投影（假设平面道路）
            # 3D世界坐标（相机坐标系）
            x_world = u_norm * depth_meters * 0.5  # 横向位置（米）
            y_world = depth_meters  # 纵向位置（米）
            
            # 转换到BEV坐标系
            # BEV中心对应自车位置，顶部是前方
            bev_x = int(bev_w / 2 + x_world / 10.0 * bev_w / 2)  # 横向范围 ±10米
            bev_y = int(bev_h - (y_world / 50.0) * bev_h)  # 纵向范围 0-50米
            
            # 边界检查
            if 0 <= bev_x < bev_w and 0 <= bev_y < bev_h:
                # 累积颜色和深度
                bev_image[bev_y, bev_x] += image_rgb[v, u]
                bev_depth[bev_y, bev_x] += depth_normalized
                bev_count[bev_y, bev_x] += 1
    
    # 平均化重叠的像素
    mask = bev_count > 0
    bev_image[mask] = (bev_image[mask] / bev_count[mask, np.newaxis]).astype(np.uint8)
    bev_depth[mask] = bev_depth[mask] / bev_count[mask]
    
    return bev_image, bev_depth

def add_bev_annotations(bev_image, bev_depth):
    """
    在BEV图像上添加标注（网格线、距离标记）
    """
    annotated = bev_image.copy()
    h, w = bev_image.shape[:2]
    
    # 绘制网格线（每10米一条横线）
    for i in range(0, 6):  # 0, 10, 20, 30, 40, 50米
        y = int(h - (i / 5.0) * h)
        cv2.line(annotated, (0, y), (w, y), (0, 255, 0), 1)
        # 添加距离标记
        text = f"{i*10}m"
        cv2.putText(annotated, text, (w - 50, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # 绘制中心线
    cv2.line(annotated, (w // 2, 0), (w // 2, h), (255, 255, 0), 2)
    
    # 标记自车位置
    ego_y = h - 10
    ego_x = w // 2
    cv2.rectangle(annotated, (ego_x - 15, ego_y - 10), 
                  (ego_x + 15, ego_y + 10), (0, 255, 255), -1)
    cv2.putText(annotated, "EGO", (ego_x - 12, ego_y + 3),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return annotated

def process_driving_image(image_path, output_dir):
    """
    处理单张驾驶图像，生成BEV可视化
    """
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # 调整大小以加快处理（可选）
    target_width = 640
    target_height = 480
    image = image.resize((target_width, target_height))
    image_np = np.array(image)
    
    print(f"  Processing: {os.path.basename(image_path)}")
    print(f"    Image size: {image_np.shape}")
    
    # 1. 深度估计
    print("    [1/3] Estimating depth...")
    depth_map = simple_depth_estimation(image_np)
    
    # 2. 生成BEV
    print("    [2/3] Generating BEV...")
    bev_image, bev_depth = depth_to_bev(image_np, depth_map)
    
    # 3. 添加标注
    print("    [3/3] Adding annotations...")
    bev_annotated = add_bev_annotations(bev_image, bev_depth)
    
    return image_np, depth_map, bev_image, bev_annotated

def create_comparison_visualization(results_list, output_path):
    """
    创建多图像的对比可视化
    """
    n_images = len(results_list)
    fig, axes = plt.subplots(n_images, 4, figsize=(20, 5*n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (orig_img, depth_map, bev_img, bev_annotated) in enumerate(results_list):
        # 第一列：原始图像
        axes[idx, 0].imshow(orig_img)
        axes[idx, 0].set_title('Original Perspective View\n(Front Camera)', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # 第二列：深度图
        depth_viz = axes[idx, 1].imshow(depth_map, cmap='turbo')
        axes[idx, 1].set_title('Estimated Depth Map\n(Closer=Red, Farther=Blue)', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')
        plt.colorbar(depth_viz, ax=axes[idx, 1], fraction=0.046, pad=0.04)
        
        # 第三列：原始BEV
        axes[idx, 2].imshow(bev_img)
        axes[idx, 2].set_title('BEV (Bird\'s Eye View)\n(Raw Projection)', fontsize=12, fontweight='bold')
        axes[idx, 2].axis('off')
        
        # 第四列：标注的BEV
        axes[idx, 3].imshow(bev_annotated)
        axes[idx, 3].set_title('BEV with Annotations\n(Grid & Distance Markers)', fontsize=12, fontweight='bold')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def main():
    """
    主函数：从真实驾驶数据生成BEV
    """
    print("=" * 70)
    print("BEV from Real Driving Data")
    print("=" * 70)
    print()
    
    # 检查数据目录
    data_dir = os.path.join(os.path.dirname(__file__), 
                           'behavioral_cloning_data', 'IMG')
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory not found: {data_dir}")
        print("Please run behavioral_cloning_download.py first.")
        return
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择测试图像（选择有代表性的图像）
    all_images = sorted([f for f in os.listdir(data_dir) if f.startswith('center_')])
    
    if len(all_images) == 0:
        print("❌ Error: No images found in data directory.")
        return
    
    print(f"Found {len(all_images)} images in dataset.")
    print(f"Selecting 3 representative images for BEV generation...\n")
    
    # 选择3张代表性图像（开头、中间、结尾）
    selected_indices = [
        0,  # 第一张
        len(all_images) // 2,  # 中间
        min(len(all_images) - 1, 100)  # 第100张或最后一张
    ]
    
    selected_images = [all_images[i] for i in selected_indices]
    
    # 处理每张图像
    results_list = []
    for img_name in selected_images:
        img_path = os.path.join(data_dir, img_name)
        result = process_driving_image(img_path, output_dir)
        results_list.append(result)
        print()
    
    # 生成对比可视化
    print("Generating comparison visualization...")
    output_path = os.path.join(output_dir, 'bev_from_real_data.png')
    create_comparison_visualization(results_list, output_path)
    
    # 生成总结报告
    print("\n" + "=" * 70)
    print("✅ BEV Generation from Real Data Completed!")
    print("=" * 70)
    print(f"\nGenerated Files:")
    print(f"  • {output_path}")
    print(f"\nVisualization Columns:")
    print(f"  1. Original Perspective View - 真实驾驶图像（透视投影）")
    print(f"  2. Estimated Depth Map - 深度估计结果（红=近，蓝=远）")
    print(f"  3. BEV Raw Projection - BEV原始投影（俯视图）")
    print(f"  4. BEV with Annotations - 带标注的BEV（网格线+距离标记）")
    print(f"\nKey Concepts:")
    print(f"  • Real driving images → Depth estimation → BEV projection")
    print(f"  • Compare with Chapter 4.2 perspective-based driving")
    print(f"  • Understand why BEV is useful for planning")
    print(f"\nNote:")
    print(f"  • This uses simplified depth estimation for educational purposes")
    print(f"  • Real systems use deep learning models (MiDaS, BEVFormer, LSS)")
    print(f"  • Single camera limitations: no multi-view fusion")
    print()

if __name__ == "__main__":
    # 检查依赖
    try:
        import cv2
    except ImportError:
        print("Error: OpenCV not installed. Please run:")
        print("  pip install opencv-python")
        exit(1)
    
    main()
