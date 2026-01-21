"""
CIFAR-10 智能诊断训练脚本（阶段1：诊断+建议，教学友好版本）
基于教程中的训练问题诊断表，自动检测训练问题并提供改进建议
"""
import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

try:
    import torchvision
    from torchvision import transforms
except ImportError as exc:
    raise SystemExit("Missing torchvision. Install with: python -m pip install torchvision") from exc


# ============================================================================
# 模型定义
# ============================================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class EpochRow:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


@dataclass
class RunMetrics:
    model: str
    num_params: int
    device: str
    num_epochs: int
    train_subset: int
    test_subset: int
    batch_size: int
    lr: float
    weight_decay: float
    data_augment: bool
    elapsed_sec: float
    final_train_acc: float
    final_val_acc: float
    final_train_loss: float
    final_val_loss: float
    diagnosed_problems: List[str]
    recommendations: List[str]


# ============================================================================
# 数据质量检查器（训练前主动策略）
# ============================================================================

class DataQualityChecker:
    """训练前数据质量检查（对应教程数据操作表-训练前部分）"""
    
    @staticmethod
    def check_data_quality(dataset, num_classes=10):
        """EDA式数据检查"""
        issues = []
        recommendations = []
        
        print("\n" + "="*60)
        print("📊 数据质量预检查（训练前主动策略）")
        print("="*60)
        
        # 检查1: 数据量是否充足
        n_samples = len(dataset)
        print(f"✓ 数据集大小: {n_samples} 样本")
        
        if n_samples < 5000:
            issues.append(f"数据量偏小 ({n_samples} < 5000)")
            recommendations.append(
                "建议1: 使用数据增强 (RandomHorizontalFlip + RandomCrop)"
            )
            print(f"  ⚠️  数据量偏小: {n_samples} < 5000")
        else:
            print(f"  ✓ 数据量充足")
        
        # 检查2: 类别是否平衡
        labels = []
        for i in range(min(1000, n_samples)):
            _, label = dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            labels.append(label)
        
        label_dist = Counter(labels)
        if len(label_dist) > 0:
            max_count = max(label_dist.values())
            min_count = min(label_dist.values())
            imbalance_ratio = max_count / max(min_count, 1)
            
            print(f"✓ 类别分布（采样{len(labels)}个）: {dict(label_dist)}")
            
            if imbalance_ratio > 3:
                issues.append(f"类别不平衡 (比例={imbalance_ratio:.2f})")
                recommendations.append(
                    "建议2: 使用加权采样或过采样/欠采样"
                )
                print(f"  ⚠️  类别不平衡: 最大/最小比例={imbalance_ratio:.2f}")
            else:
                print(f"  ✓ 类别分布平衡")
        
        # 检查3: 图像统计量（用于归一化）
        import numpy as np
        sample_images = []
        for i in range(min(100, n_samples)):
            img, _ = dataset[i]
            if isinstance(img, torch.Tensor):
                sample_images.append(img.numpy())
        
        if sample_images:
            sample_array = np.array(sample_images)
            mean = sample_array.mean(axis=(0, 2, 3))
            std = sample_array.std(axis=(0, 2, 3))
            
            print(f"✓ 图像统计量（采样{len(sample_images)}张）:")
            print(f"  - 均值: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
            print(f"  - 标准差: [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
        
        # 总结
        print("\n" + "-"*60)
        if len(issues) == 0:
            print("✅ 数据质量检查通过，可以开始训练")
        else:
            print(f"⚠️  发现 {len(issues)} 个潜在问题:")
            for issue in issues:
                print(f"   • {issue}")
            print(f"\n💡 改进建议:")
            for rec in recommendations:
                print(f"   • {rec}")
        print("="*60 + "\n")
        
        return {
            'n_samples': n_samples,
            'issues': issues,
            'recommendations': recommendations
        }


# ============================================================================
# 训练诊断器（训练后响应式检查）
# ============================================================================

class TrainingDiagnostic:
    """训练问题自动诊断器（对应教程训练问题诊断表）"""
    
    def __init__(self):
        self.history = []
        self.all_problems = []
        self.all_recommendations = []
    
    def add_epoch(self, metrics: dict):
        """添加一个epoch的记录"""
        self.history.append(metrics)
    
    def diagnose_current_state(self, epoch: int) -> Tuple[List[str], List[str]]:
        """诊断当前训练状态，返回(问题列表, 建议列表)"""
        problems = []
        recommendations = []
        
        if len(self.history) == 0:
            return problems, recommendations
        
        last = self.history[-1]
        
        # 问题1: Loss不下降/不收敛（5个epoch后loss变化<1%）
        if self._check_convergence_stall():
            problems.append("Loss不下降/不收敛")
            recommendations.append("★ 建议: 学习率降至0.1倍（如1e-3→1e-4）")
        
        # 问题2: 过拟合严重（train-val gap>30%）
        if self._check_overfitting():
            problems.append("过拟合严重")
            gap = last['train_acc'] - last['val_acc']
            recommendations.append(
                f"★ 建议: L2正则化(weight_decay=1e-4)或数据增强 (当前gap={gap:.1%})"
            )
        
        # 问题3: Loss震荡/不稳定（相邻epoch波动>20%）
        if self._check_instability():
            problems.append("Loss震荡/不稳定")
            recommendations.append("★ 建议: 学习率降至0.5倍（如1e-3→5e-4）")
        
        # 问题4: 收敛太慢（10个epoch后val_acc增长<5%）
        if self._check_slow_convergence():
            problems.append("收敛太慢")
            recommendations.append("★ 建议: 学习率增至2倍（如1e-4→2e-4）")
        
        # 问题5: 欠拟合（train_acc<70% 且 val_acc<65%）
        if self._check_underfitting():
            problems.append("欠拟合")
            recommendations.append(
                "★ 建议: 增加模型容量（层数+2或通道数翻倍）或检查数据量"
            )
        
        # 问题6: NaN/Inf
        if self._check_nan_inf(last):
            problems.append("出现NaN/Inf")
            recommendations.append("★ 建议: 学习率降至1e-5或使用梯度裁剪")
        
        return problems, recommendations
    
    def _check_convergence_stall(self) -> bool:
        """检查：5个epoch后loss变化<1%"""
        if len(self.history) < 5:
            return False
        recent_losses = [h['train_loss'] for h in self.history[-5:]]
        max_loss = max(recent_losses)
        min_loss = min(recent_losses)
        if max_loss < 1e-8:
            return False
        loss_change = (max_loss - min_loss) / max_loss
        return loss_change < 0.01
    
    def _check_overfitting(self) -> bool:
        """检查：train-val gap>30%"""
        if not self.history:
            return False
        last = self.history[-1]
        gap = last['train_acc'] - last['val_acc']
        return gap > 0.30
    
    def _check_instability(self) -> bool:
        """检查：相邻epoch波动>20%"""
        if len(self.history) < 2:
            return False
        recent = self.history[-2:]
        loss_change = abs(recent[1]['train_loss'] - recent[0]['train_loss'])
        if recent[0]['train_loss'] < 1e-8:
            return False
        volatility = loss_change / recent[0]['train_loss']
        return volatility > 0.20
    
    def _check_slow_convergence(self) -> bool:
        """检查：10个epoch后val_acc增长<5%"""
        if len(self.history) < 10:
            return False
        acc_gain = self.history[-1]['val_acc'] - self.history[-10]['val_acc']
        return acc_gain < 0.05
    
    def _check_underfitting(self) -> bool:
        """检查：train_acc<70% 且 val_acc<65%"""
        if not self.history:
            return False
        last = self.history[-1]
        return last['train_acc'] < 0.70 and last['val_acc'] < 0.65
    
    def _check_nan_inf(self, metrics: dict) -> bool:
        """检查：是否出现NaN/Inf"""
        import math
        for v in metrics.values():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return True
        return False
    
    def print_diagnosis(self, epoch: int):
        """打印诊断结果"""
        problems, recommendations = self.diagnose_current_state(epoch)
        
        if problems:
            print("\n" + "="*60)
            print(f"⚠️  训练诊断报告 (Epoch {epoch})")
            print("="*60)
            print(f"检测到 {len(problems)} 个问题:")
            for i, prob in enumerate(problems, 1):
                print(f"  {i}. {prob}")
            
            print(f"\n💡 改进建议:")
            for rec in recommendations:
                print(f"  {rec}")
            print("="*60 + "\n")
            
            # 记录所有问题和建议（用于最终报告）
            for p in problems:
                if p not in self.all_problems:
                    self.all_problems.append(p)
            for r in recommendations:
                if r not in self.all_recommendations:
                    self.all_recommendations.append(r)


# ============================================================================
# 诊断仪表盘可视化
# ============================================================================

class TrainingDashboard:
    """实时训练诊断仪表盘"""
    
    @staticmethod
    def plot_diagnostic_dashboard(history: List[dict], output_path: str):
        """生成诊断仪表盘"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  matplotlib未安装，跳过仪表盘生成")
            return
        
        if not history:
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        val_loss = [h['val_loss'] for h in history]
        train_acc = [h['train_acc'] for h in history]
        val_acc = [h['val_acc'] for h in history]
        
        # 1. Loss曲线
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
        ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确率曲线
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
        ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 过拟合指标
        ax3 = fig.add_subplot(gs[0, 2])
        overfitting = [h['train_acc'] - h['val_acc'] for h in history]
        ax3.plot(epochs, overfitting, 'orange', linewidth=2)
        ax3.axhline(y=0.30, color='r', linestyle='--', label='Threshold (30%)', linewidth=1)
        ax3.set_title('Overfitting Gap', fontsize=12)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Train-Val Gap')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. Loss稳定性
        ax4 = fig.add_subplot(gs[1, 2])
        if len(train_loss) > 1:
            loss_volatility = [
                abs(train_loss[i] - train_loss[i-1]) / max(train_loss[i-1], 1e-8)
                for i in range(1, len(train_loss))
            ]
            ax4.plot(epochs[1:], loss_volatility, 'purple', linewidth=2)
            ax4.axhline(y=0.20, color='r', linestyle='--', label='Threshold (20%)', linewidth=1)
        ax4.set_title('Loss Volatility', fontsize=12)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Relative Change')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. 诊断文本框
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # 自动诊断
        last = history[-1]
        diagnosis_text = "🔍 自动诊断结果:\n\n"
        
        if last['train_acc'] - last['val_acc'] > 0.30:
            gap = last['train_acc'] - last['val_acc']
            diagnosis_text += f"⚠️  过拟合严重 (gap={gap:.1%} > 30%)\n"
            diagnosis_text += "   → 建议：增加weight_decay或数据增强\n\n"
        
        if len(history) >= 5:
            recent_losses = [h['train_loss'] for h in history[-5:]]
            max_loss = max(recent_losses)
            min_loss = min(recent_losses)
            if max_loss > 1e-8:
                change_rate = (max_loss - min_loss) / max_loss
                if change_rate < 0.01:
                    diagnosis_text += f"⚠️  Loss不收敛 (5 epoch变化={change_rate:.1%} < 1%)\n"
                    diagnosis_text += "   → 建议：降低学习率至0.1倍\n\n"
        
        if last['train_acc'] < 0.70 and last['val_acc'] < 0.65:
            diagnosis_text += "⚠️  欠拟合\n"
            diagnosis_text += "   → 建议：增加模型容量或提高学习率\n\n"
        
        if "⚠️" not in diagnosis_text:
            diagnosis_text += "✅ 训练状态正常\n"
        
        ax5.text(0.05, 0.5, diagnosis_text, 
                fontsize=11, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存诊断仪表盘: {output_path}")


# ============================================================================
# 训练与评估函数
# ============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / len(loader)
    acc = correct / max(total, 1)
    return avg_loss, acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(loader)
    acc = correct / max(total, 1)
    return avg_loss, acc


# ============================================================================
# 工具函数
# ============================================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cifar10_transforms(data_augment: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tfms: List[transforms.Compose] = []
    if data_augment:
        train_tfms.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    train_tfms.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(train_tfms), test_tfms


def _save_train_log_csv(rows: List[EpochRow], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _plot_training(rows: List[EpochRow], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r.epoch for r in rows]
    train_loss = [r.train_loss for r in rows]
    val_loss = [r.val_loss for r in rows]
    train_acc = [r.train_acc for r in rows]
    val_acc = [r.val_acc for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, train_loss, label="train_loss", linewidth=2)
    ax1.plot(epochs, val_loss, label="val_loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_title("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, train_acc, label="train_acc", linewidth=2)
    ax2.plot(epochs, val_acc, label="val_acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_title("Accuracy")
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# 智能训练主函数
# ============================================================================

def run_experiment_with_diagnosis(
    *,
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    train_dataset,
    device: torch.device,
    out_dir: str,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    data_augment: bool,
    train_subset: int,
    test_subset: int,
    batch_size: int,
) -> Dict:
    """带智能诊断的训练函数"""
    _ensure_dir(out_dir)
    
    # 训练前数据质量检查
    data_quality = DataQualityChecker.check_data_quality(train_dataset)
    
    # 初始化诊断器
    diagnostic = TrainingDiagnostic()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[EpochRow] = []
    start = time.time()

    print(f"\n{'='*60}")
    print(f"🚀 开始训练: {name}")
    print(f"{'='*60}")
    print(f"模型参数量: {_count_params(model):,}")
    print(f"训练样本: {train_subset}, 验证样本: {test_subset}")
    print(f"学习率: {lr}, Weight Decay: {weight_decay}, 数据增强: {data_augment}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, test_loader, criterion, device)
        
        epoch_row = EpochRow(
            epoch=epoch + 1,
            train_loss=round(train_loss, 6),
            train_acc=round(train_acc, 6),
            val_loss=round(val_loss, 6),
            val_acc=round(val_acc, 6),
        )
        history.append(epoch_row)
        
        # 添加到诊断器
        diagnostic.add_epoch({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        # 打印进度
        gap = train_acc - val_acc
        print(
            f"Epoch {epoch+1:2d}/{num_epochs} | "
            f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f} | "
            f"gap: {gap:+.1%}"
        )
        
        # 每3个epoch进行一次诊断（避免过于频繁）
        if (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
            diagnostic.print_diagnosis(epoch + 1)

    elapsed = time.time() - start
    num_params = _count_params(model)

    # 保存标准输出
    train_log_path = os.path.join(out_dir, "train_log.csv")
    model_path = os.path.join(out_dir, "model.pth")
    plot_path = os.path.join(out_dir, "training_plot.png")
    dashboard_path = os.path.join(out_dir, "diagnostic_dashboard.png")
    metrics_path = os.path.join(out_dir, "metrics.json")

    _save_train_log_csv(history, train_log_path)
    torch.save(model.state_dict(), model_path)
    _plot_training(history, plot_path)
    
    # 生成诊断仪表盘
    TrainingDashboard.plot_diagnostic_dashboard(diagnostic.history, dashboard_path)

    last = history[-1]
    metrics = RunMetrics(
        model=name,
        num_params=num_params,
        device=str(device),
        num_epochs=num_epochs,
        train_subset=train_subset,
        test_subset=test_subset,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        data_augment=data_augment,
        elapsed_sec=round(elapsed, 2),
        final_train_acc=last.train_acc,
        final_val_acc=last.val_acc,
        final_train_loss=last.train_loss,
        final_val_loss=last.val_loss,
        diagnosed_problems=diagnostic.all_problems,
        recommendations=diagnostic.all_recommendations,
    )
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)

    # 打印最终总结
    print("\n" + "="*60)
    print("✅ 训练完成！")
    print("="*60)
    print(f"总用时: {elapsed:.2f}秒")
    print(f"最终训练准确率: {last.train_acc:.4f}")
    print(f"最终验证准确率: {last.val_acc:.4f}")
    print(f"过拟合程度: {(last.train_acc - last.val_acc):.1%}")
    
    if diagnostic.all_problems:
        print(f"\n⚠️  训练过程中检测到的问题:")
        for prob in diagnostic.all_problems:
            print(f"   • {prob}")
        print(f"\n💡 总体改进建议:")
        for rec in diagnostic.all_recommendations:
            print(f"   • {rec}")
    else:
        print("\n✅ 训练过程未检测到明显问题")
    
    print("\n📁 输出文件:")
    print(f"   • 训练日志: {train_log_path}")
    print(f"   • 模型权重: {model_path}")
    print(f"   • 训练曲线: {plot_path}")
    print(f"   • 诊断仪表盘: {dashboard_path}")
    print(f"   • 指标摘要: {metrics_path}")
    print("="*60 + "\n")

    return {
        "out_dir": out_dir,
        "train_log": train_log_path,
        "model_path": model_path,
        "training_plot": plot_path,
        "diagnostic_dashboard": dashboard_path,
        "metrics": metrics_path,
        "metrics_obj": asdict(metrics),
        "history": [asdict(r) for r in history],
        "data_quality": data_quality,
    }


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 智能诊断训练（阶段1：诊断+建议）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simple_cnn",
        choices=["simple_cnn", "resnet18", "both"],
        help="Which model to train",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--subset", type=int, default=5000, help="Train subset size")
    parser.add_argument("--test_subset", type=int, default=1000, help="Test subset size")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--data_augment",
        type=str,
        default="False",
        help="True/False. Enable basic data augmentation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for this experiment",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_augment = str(args.data_augment).lower() in {"1", "true", "yes", "y"}

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_out_root = os.path.join(base_dir, "tutorial_runs", "output")
    out_dir = args.output or os.path.join(default_out_root, f"{args.model}_smart")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(args.seed)

    train_tfm, test_tfm = _cifar10_transforms(data_augment=data_augment)
    data_root = os.path.join(base_dir, "tutorial_runs", "data")
    train_ds = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_tfm,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_tfm,
    )

    train_subset = Subset(train_ds, list(range(min(args.subset, len(train_ds)))))
    test_subset = Subset(test_ds, list(range(min(args.test_subset, len(test_ds)))))
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    def _make_model(model_name: str) -> nn.Module:
        if model_name == "simple_cnn":
            return SimpleCNN(num_classes=10)
        if model_name == "resnet18":
            return torchvision.models.resnet18(weights=None, num_classes=10)
        raise ValueError(f"Unknown model: {model_name}")

    runs: List[Dict] = []
    models_to_run: List[str]
    if args.model == "both":
        models_to_run = ["simple_cnn", "resnet18"]
    else:
        models_to_run = [args.model]

    for model_name in models_to_run:
        this_out = out_dir
        if args.model == "both":
            this_out = os.path.join(out_dir, model_name)

        model = _make_model(model_name).to(device)
        run = run_experiment_with_diagnosis(
            name=model_name,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            train_dataset=train_subset,
            device=device,
            out_dir=this_out,
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            data_augment=data_augment,
            train_subset=len(train_subset),
            test_subset=len(test_subset),
            batch_size=args.batch_size,
        )
        runs.append(run)

    # 如果运行多个模型，生成对比报告
    if len(runs) > 1:
        _ensure_dir(default_out_root)
        summary_path = os.path.join(default_out_root, "smart_training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "model": r["metrics_obj"]["model"],
                        "metrics": r["metrics_obj"],
                        "data_quality": r["data_quality"],
                        "out_dir": r["out_dir"],
                    }
                    for r in runs
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"✓ 对比摘要已保存: {summary_path}")


if __name__ == "__main__":
    main()
