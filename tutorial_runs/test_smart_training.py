"""
快速测试智能诊断训练系统
运行3个epoch快速验证功能
"""
import subprocess
import sys
import os

def run_test():
    print("="*60)
    print("智能诊断训练系统 - 快速测试")
    print("="*60)
    print()
    
    # 获取脚本路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "cifar_cnn_resnet_smart.py")
    
    # 测试命令（3个epoch，快速验证）
    cmd = [
        sys.executable,
        script_path,
        "--model", "simple_cnn",
        "--epochs", "3",
        "--subset", "1000",  # 小数据集，触发数据量警告
        "--test_subset", "200",
        "--output", os.path.join(script_dir, "output", "test_smart")
    ]
    
    print("🚀 运行测试命令:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("="*60)
        print("✅ 测试成功完成！")
        print("="*60)
        print()
        print("📁 检查输出文件:")
        output_dir = os.path.join(script_dir, "output", "test_smart")
        for filename in ["train_log.csv", "model.pth", "training_plot.png", 
                        "diagnostic_dashboard.png", "metrics.json"]:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename} (缺失)")
        print()
        print("💡 提示:")
        print(f"  查看诊断仪表盘: {os.path.join(output_dir, 'diagnostic_dashboard.png')}")
        print(f"  查看指标摘要: {os.path.join(output_dir, 'metrics.json')}")
        
    except subprocess.CalledProcessError as e:
        print()
        print("="*60)
        print("❌ 测试失败")
        print("="*60)
        print(f"错误代码: {e.returncode}")
        return False
    except Exception as e:
        print()
        print("="*60)
        print(f"❌ 发生错误: {e}")
        print("="*60)
        return False
    
    return True

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
