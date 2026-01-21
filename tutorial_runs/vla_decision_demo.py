"""
VLA (Vision-Language-Action) Decision Demo
演示语言引导的驾驶决策 vs 纯视觉决策的对比

目的：让学员理解VLA如何通过语言指令影响决策
方法：CLIP特征提取 + 模拟策略网络
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

# 检查CLIP依赖
try:
    import open_clip
    import torch
except ImportError:
    print("Error: Missing dependencies. Please install:")
    print("  pip install open_clip_torch torch")
    exit(1)

class SimpleVLADecisionSimulator:
    """
    简化的VLA决策模拟器
    使用CLIP特征 + 随机初始化的策略头
    
    注意：这是教学演示，不是真实训练的VLA模型
    真实VLA需要大规模数据和长时间训练
    """
    
    def __init__(self):
        print("Initializing VLA Decision Simulator...")
        
        # 加载CLIP模型（用于特征提取）
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', 
            pretrained='openai'
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        # 模拟策略网络头（随机初始化，仅用于演示）
        # 真实VLA会通过大量数据训练这个策略头
        torch.manual_seed(42)  # 固定随机种子，保证可复现
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3)  # 输出: [steering, throttle, brake]
        ).to(self.device)
        
        # 语言条件策略头（用于处理文本特征）
        self.language_policy_head = torch.nn.Sequential(
            torch.nn.Linear(1024, 256),  # 拼接视觉+文本特征
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3)
        ).to(self.device)
        
        print("✅ VLA Simulator initialized successfully")
    
    def encode_image(self, image_path):
        """提取图像特征（使用CLIP视觉编码器）"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def encode_text(self, text):
        """提取文本特征（使用CLIP文本编码器）"""
        text_tokens = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def predict_vision_only(self, image_path):
        """纯视觉决策（类似行为克隆）"""
        image_features = self.encode_image(image_path)
        
        with torch.no_grad():
            action = self.policy_head(image_features)
            action = torch.tanh(action)  # 归一化到[-1, 1]
        
        return {
            'steering': action[0, 0].item(),
            'throttle': action[0, 1].item(),
            'brake': action[0, 2].item()
        }
    
    def predict_language_guided(self, image_path, instruction):
        """语言引导决策（VLA）"""
        image_features = self.encode_image(image_path)
        text_features = self.encode_text(instruction)
        
        # 多模态特征融合（简单拼接）
        fused_features = torch.cat([image_features, text_features], dim=-1)
        
        with torch.no_grad():
            action = self.language_policy_head(fused_features)
            action = torch.tanh(action)
        
        return {
            'steering': action[0, 0].item(),
            'throttle': action[0, 1].item(),
            'brake': action[0, 2].item()
        }
    
    def compare_decisions(self, image_path, instructions):
        """对比不同决策方式"""
        results = {
            'image': os.path.basename(image_path),
            'vision_only': self.predict_vision_only(image_path),
            'language_guided': {}
        }
        
        for instruction in instructions:
            results['language_guided'][instruction] = self.predict_language_guided(
                image_path, instruction
            )
        
        return results

def create_comparison_visualization(results_list, output_path):
    """创建决策对比可视化"""
    n_images = len(results_list)
    fig, axes = plt.subplots(n_images, 4, figsize=(20, 5*n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(results_list):
        # 读取图像
        image_name = result['image']
        image_path = os.path.join(
            os.path.dirname(__file__), 
            'behavioral_cloning_data', 'IMG', image_name
        )
        
        try:
            img = Image.open(image_path)
        except:
            img = Image.new('RGB', (320, 160), color='gray')
        
        # 第一列：原始图像
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'Image: {image_name[:30]}...', fontsize=10)
        axes[idx, 0].axis('off')
        
        # 第二列：纯视觉决策
        vision_action = result['vision_only']
        axes[idx, 1].barh(['Steering', 'Throttle', 'Brake'], 
                          [vision_action['steering'], vision_action['throttle'], vision_action['brake']],
                          color=['blue', 'green', 'red'])
        axes[idx, 1].set_xlim(-1, 1)
        axes[idx, 1].set_title('Vision-Only Decision\n(Like Behavioral Cloning)', fontsize=10, fontweight='bold')
        axes[idx, 1].axvline(x=0, color='black', linestyle='--', alpha=0.3)
        
        # 第三列和第四列：语言引导决策
        col_offset = 2
        for instr_idx, (instruction, action) in enumerate(result['language_guided'].items()):
            if instr_idx >= 2:  # 最多显示2个指令
                break
            
            axes[idx, col_offset + instr_idx].barh(
                ['Steering', 'Throttle', 'Brake'],
                [action['steering'], action['throttle'], action['brake']],
                color=['purple', 'orange', 'pink']
            )
            axes[idx, col_offset + instr_idx].set_xlim(-1, 1)
            axes[idx, col_offset + instr_idx].set_title(
                f'Language-Guided:\n"{instruction}"', 
                fontsize=10, fontweight='bold'
            )
            axes[idx, col_offset + instr_idx].axvline(x=0, color='black', linestyle='--', alpha=0.3)
        
        # 添加行标签
        if idx == 0:
            fig.text(0.01, 0.5, 'Comparison Results', va='center', rotation='vertical',
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization: {output_path}")
    plt.close()

def create_summary_report(results_list, output_path):
    """生成文字总结报告"""
    report = "VLA Decision Demo - Summary Report\n"
    report += "=" * 70 + "\n\n"
    
    for idx, result in enumerate(results_list, 1):
        report += f"Image {idx}: {result['image']}\n"
        report += "-" * 70 + "\n"
        
        # 纯视觉决策
        vision = result['vision_only']
        report += f"Vision-Only Decision:\n"
        report += f"  Steering: {vision['steering']:+.3f}\n"
        report += f"  Throttle: {vision['throttle']:+.3f}\n"
        report += f"  Brake:    {vision['brake']:+.3f}\n\n"
        
        # 语言引导决策
        for instruction, action in result['language_guided'].items():
            report += f"Language-Guided Decision (\"{instruction}\"):\n"
            report += f"  Steering: {action['steering']:+.3f}\n"
            report += f"  Throttle: {action['throttle']:+.3f}\n"
            report += f"  Brake:    {action['brake']:+.3f}\n"
            
            # 计算差异
            delta_steering = action['steering'] - vision['steering']
            report += f"  → Steering Δ: {delta_steering:+.3f} (impact of language)\n\n"
        
        report += "\n"
    
    report += "=" * 70 + "\n"
    report += "Key Insights:\n"
    report += "- Vision-only decision: Depends solely on visual features\n"
    report += "- Language-guided decision: Influenced by natural language instructions\n"
    report += "- Delta values show how language modifies the decision\n\n"
    report += "Note: This is a SIMULATED demo using random policy heads.\n"
    report += "Real VLA models require large-scale training on image-text-action datasets.\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Saved summary report: {output_path}")
    return report

def main():
    """主函数：运行VLA决策演示"""
    print("=" * 70)
    print("VLA (Vision-Language-Action) Decision Demo")
    print("=" * 70)
    print()
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化VLA模拟器
    simulator = SimpleVLADecisionSimulator()
    
    print("\n[1/3] Running decision comparisons...")
    
    # 选择测试图像
    data_dir = os.path.join(os.path.dirname(__file__), 'behavioral_cloning_data', 'IMG')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Please run behavioral_cloning_download.py first.")
        return
    
    # 选择几张代表性图像
    test_images = []
    all_images = sorted([f for f in os.listdir(data_dir) if f.startswith('center_')])
    
    if len(all_images) == 0:
        print("Error: No images found in data directory.")
        return
    
    # 选择3张图像作为示例
    indices = [0, len(all_images)//2, -1]
    for idx in indices:
        if idx < len(all_images):
            test_images.append(os.path.join(data_dir, all_images[idx]))
    
    # 定义测试指令
    instructions = [
        "turn left carefully",
        "turn right at intersection"
    ]
    
    # 运行对比实验
    results_list = []
    for image_path in test_images[:3]:  # 最多3张图像
        print(f"  Processing: {os.path.basename(image_path)}")
        result = simulator.compare_decisions(image_path, instructions)
        results_list.append(result)
    
    print("\n[2/3] Generating visualizations...")
    
    # 生成可视化
    viz_path = os.path.join(output_dir, 'vla_decision_comparison.png')
    create_comparison_visualization(results_list, viz_path)
    
    # 保存JSON结果
    json_path = os.path.join(output_dir, 'vla_decision_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2)
    print(f"✅ Saved JSON results: {json_path}")
    
    print("\n[3/3] Generating summary report...")
    
    # 生成总结报告
    report_path = os.path.join(output_dir, 'vla_decision_summary.txt')
    report = create_summary_report(results_list, report_path)
    
    print("\n" + "=" * 70)
    print("✅ VLA Decision Demo completed successfully!")
    print("=" * 70)
    print("\nGenerated Files:")
    print(f"  • {viz_path}")
    print(f"  • {json_path}")
    print(f"  • {report_path}")
    print("\nKey Concepts Demonstrated:")
    print("  1. Vision-Only Decision: Pure visual input → action (like behavioral cloning)")
    print("  2. Language-Guided Decision: Visual + text instruction → action (VLA)")
    print("  3. Decision Difference: How language modifies the action")
    print("\nEducational Value:")
    print("  • Understand VLA's multimodal fusion")
    print("  • See language's impact on decision-making")
    print("  • Compare with Chapter 4.2 behavioral cloning")
    print("\nImportant Note:")
    print("  This is a SIMULATED demo using random policy networks.")
    print("  Real VLA models (like RT-2, OpenVLA) require:")
    print("    - Millions of training examples")
    print("    - Hours/days of GPU training")
    print("    - Specialized datasets with image-text-action triplets")
    print()
    
    # 打印部分报告
    print("\n" + "-" * 70)
    print("Sample Results:")
    print("-" * 70)
    print(report.split('\n\n')[1])  # 打印第一个图像的结果

if __name__ == "__main__":
    main()
