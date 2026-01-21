#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在第3章理论部分添加对应第4章实操案例的提及 - 修正版
"""

def main():
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("开始添加实操案例引用...")
    
    # 更新1: 3.2.2 添加行为克隆引用
    old1 = '本课程实操不强求复杂世界模型实现，但要求理解"为什么必须预测"。'
    new1 = old1 + '\n\n**对应实操**：4.2节行为克隆展示最简单的感知-动作映射（无预测）。'
    if old1 in content:
        # 只替换第一次出现
        pos = content.find(old1)
        content = content[:pos] + new1 + content[pos+len(old1):]
        print("✓ 已添加 3.2.2 → 4.2 行为克隆引用")
    else:
        print("✗ 未找到 3.2.2 匹配文本")
    
    # 更新2: 3.2.3 添加世界模型引用
    old2 = '- 李飞飞的世界模型倡议：推动"可生成、可预测"的世界表示'
    new2 = old2 + '\n\n**对应实操**：4.3节Moving MNIST展示简化的时序预测（视频帧预测）。'
    if old2 in content:
        pos = content.find(old2)
        content = content[:pos] + new2 + content[pos+len(old2):]
        print("✓ 已添加 3.2.3 → 4.3 世界模型引用")
    else:
        print("✗ 未找到 3.2.3 匹配文本")
    
    # 更新3: 3.4.2 BEV 添加引用
    old3 = '扩展：李飞飞倡导的"世界模型"（World Model / World Lab）旨在学习通用、可预测的三维世界表示，这与 BEV/3D 是互补且相辅相成的。'
    new3 = old3 + '\n\n**对应实操**：4.4节展示从真实驾驶图像生成BEV俯视图的完整流程。'
    if old3 in content:
        pos = content.find(old3)
        content = content[:pos] + new3 + content[pos+len(old3):]
        print("✓ 已添加 3.4.2 → 4.4 BEV引用")
    else:
        print("✗ 未找到 3.4.2 匹配文本")
    
    # 更新4: 3.4.3 世界模型时序建模添加引用
    old4 = '世界模型时序建模的目标是预测状态随时间变化的规律。它的重要性在于让系统不仅"看到当前"，还能"预见未来"。本课程的核心实操就是这一点：用前 T 帧预测下一帧。'
    new4 = old4 + '\n\n**对应实操**：4.3节用LSTM预测Moving MNIST视频的未来10帧。'
    if old4 in content:
        pos = content.find(old4)
        content = content[:pos] + new4 + content[pos+len(old4):]
        print("✓ 已添加 3.4.3 → 4.3 时序建模引用")
    else:
        print("✗ 未找到 3.4.3 匹配文本")
    
    # 更新6: 3.4.5 CLIP 添加引用
    old6 = 'CLIP 是视觉-语言对齐的代表模型，让模型学会"图像与文本如何对应"。在智能驾驶中可用于高层语义理解与提示。'
    new6 = old6 + '\n\n**对应实操**：4.5节VLA决策演示中使用CLIP作为多模态编码器。'
    if old6 in content:
        pos = content.find(old6)
        content = content[:pos] + new6 + content[pos+len(old6):]
        print("✓ 已添加 3.4.5 → 4.5 CLIP引用")
    else:
        print("✗ 未找到 3.4.5 匹配文本")
    
    # 写回文件
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n" + "="*60)
    print("✅ 第3章理论与第4章实操关联完成！")
    print("="*60)
    print("\n添加的关联映射：")
    print("  3.2.2 智能驾驶直接映射 → 4.2 行为克隆")
    print("  3.2.3 世界模型发展脉络 → 4.3 Moving MNIST")
    print("  3.4.2 BEV/3D表示 → 4.4 BEV空间表示")
    print("  3.4.3 世界模型时序建模 → 4.3 Moving MNIST")
    print("  3.4.4 VLM/VLA → 4.5 VLA决策 (已存在)")
    print("  3.4.5 CLIP → 4.5 VLA决策")
    print("\n每处引用不超过50字，方便学员知行合一！")

if __name__ == '__main__':
    main()
