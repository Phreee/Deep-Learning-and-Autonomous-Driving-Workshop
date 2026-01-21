"""
生成驾驶数据的GIF动画
"""
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 数据路径
data_dir = os.path.dirname(__file__)
csv_file = os.path.join(data_dir, "behavioral_cloning_data/driving_log_local.csv")
output_dir = os.path.join(data_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# 读取数据
print("读取数据...")
df = pd.read_csv(csv_file)

# 统一列名
if 'image' in df.columns:
    df['center'] = df['image']

print(f"总数据量: {len(df)} 条")

# 生成GIF动画参数
gif_params = {
    'num_frames': 100,  # GIF帧数
    'skip_frames': 5,   # 每隔多少帧采样一次（加快速度）
    'fps': 10,          # 每秒帧数
    'start_idx': 100    # 从第100帧开始（跳过初始静止部分）
}

print(f"\n生成GIF动画参数:")
print(f"  采样帧数: {gif_params['num_frames']}")
print(f"  跳帧间隔: {gif_params['skip_frames']}")
print(f"  播放速度: {gif_params['fps']} FPS")
print(f"  起始帧: {gif_params['start_idx']}")

# 生成GIF帧
print("\n开始生成GIF帧...")
frames = []
frame_indices = []

for i in range(gif_params['num_frames']):
    idx = gif_params['start_idx'] + i * gif_params['skip_frames']
    if idx >= len(df):
        break
    
    frame_indices.append(idx)
    row = df.iloc[idx]
    
    # 读取图像
    img_path = row['center'].strip()
    if not os.path.exists(img_path):
        img_path = os.path.join(data_dir, 'behavioral_cloning_data', 'IMG', os.path.basename(img_path))
    
    try:
        img = Image.open(img_path)
        
        # 调整大小
        target_width = 320
        aspect_ratio = img.height / img.width
        target_height = int(target_width * aspect_ratio)
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # 创建带信息的画布
        canvas_height = target_height + 80  # 增加底部空间显示信息
        canvas = Image.new('RGB', (target_width, canvas_height), color='white')
        canvas.paste(img, (0, 0))
        
        # 绘制信息
        draw = ImageDraw.Draw(canvas)
        steering = row['steering']
        
        # 尝试使用系统字体，如果失败则使用默认字体
        try:
            font_large = ImageFont.truetype("arial.ttf", 20)
            font_small = ImageFont.truetype("arial.ttf", 14)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # 绘制文本背景
        draw.rectangle([0, target_height, target_width, canvas_height], fill='#2c3e50')
        
        # 绘制转向信息
        steering_text = f"转向: {steering:+.3f}"
        draw.text((10, target_height + 10), steering_text, fill='white', font=font_large)
        
        # 绘制转向方向指示
        if abs(steering) < 0.05:
            direction = "直行 ↑"
            color = 'lime'
        elif steering > 0:
            direction = f"右转 →"
            color = 'red'
        else:
            direction = f"左转 ←"
            color = 'cyan'
        
        draw.text((10, target_height + 40), direction, fill=color, font=font_large)
        
        # 绘制帧信息
        frame_text = f"帧: {idx}/{len(df)}"
        draw.text((target_width - 120, target_height + 10), frame_text, fill='lightgray', font=font_small)
        
        # 绘制转向强度条
        bar_width = int(abs(steering) * 100)
        bar_color = 'red' if steering > 0 else 'blue'
        bar_y = target_height + 65
        draw.rectangle([target_width - 120, bar_y, target_width - 120 + bar_width, bar_y + 10], 
                      fill=bar_color)
        
        frames.append(canvas)
        
        if (i + 1) % 20 == 0:
            print(f"  已处理 {i + 1}/{gif_params['num_frames']} 帧")
    
    except Exception as e:
        print(f"  警告: 跳过帧 {idx}, 错误: {e}")
        continue

if len(frames) == 0:
    print("错误: 未能生成任何帧")
    exit(1)

print(f"\n成功生成 {len(frames)} 帧")

# 保存GIF
print("\n保存GIF文件...")
gif_path = os.path.join(output_dir, "driving_demo.gif")

# 计算持续时间（毫秒）
duration = int(1000 / gif_params['fps'])

frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=duration,
    loop=0,  # 无限循环
    optimize=False
)

print(f"✓ 已保存GIF动画: {gif_path}")
print(f"  文件大小: {os.path.getsize(gif_path) / 1024 / 1024:.2f} MB")
print(f"  总帧数: {len(frames)}")
print(f"  持续时间: {len(frames) / gif_params['fps']:.1f} 秒")

# 生成一个更快的版本
print("\n生成快速版本...")
fast_gif_path = os.path.join(output_dir, "driving_demo_fast.gif")
frames[0].save(
    fast_gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=50,  # 20 FPS
    loop=0,
    optimize=False
)
print(f"✓ 已保存快速版GIF: {fast_gif_path}")

print("\n" + "="*60)
print("GIF动画生成完成！")
print("="*60)
print("\n生成的文件:")
print(f"1. 标准速度 (10 FPS): driving_demo.gif")
print(f"2. 快速播放 (20 FPS): driving_demo_fast.gif")
print("\n这些GIF可以直接插入到教程文档中，让学员看到实际的驾驶场景。")
