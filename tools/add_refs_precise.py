#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在第3章理论部分添加对应第4章实操案例的提及 - 精确行号版
"""

def main():
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"文件总行数: {len(lines)}")
    print("开始添加实操案例引用...")
    
    # 先找到所有目标行号
    targets = {
        '3.2.2': None,
        '3.2.3': None,
        '3.4.3': None,
        '3.4.5': None,
        '3.4.2_BEV': None
    }
    
    for i, line in enumerate(lines):
        if '本课程实操不强求复杂世界模型实现' in line:
            targets['3.2.2'] = i + 1  # 行号从1开始
            print(f"找到 3.2.2 目标行: {i+1}")
        elif '李飞飞的世界模型倡议' in line:
            targets['3.2.3'] = i + 1
            print(f"找到 3.2.3 目标行: {i+1}")
        elif '世界模型时序建模的目标是预测状态随时间变化的规律' in line:
            targets['3.4.3'] = i + 1
            print(f"找到 3.4.3 目标行: {i+1}")
        elif 'CLIP 是视觉-语言对齐的代表模型，让模型学会"图像与文本如何对应"' in line:
            targets['3.4.5'] = i + 1
            print(f"找到 3.4.5 目标行: {i+1}")
        elif '扩展：李飞飞倡导的"世界模型"（World Model / World Lab）旨在学习通用' in line:
            # 从这里开始，找到段落结尾
            for j in range(i, min(i+10, len(lines))):
                if '#### 3.4.3' in lines[j]:
                    targets['3.4.2_BEV'] = j  # 在3.4.3之前插入
                    print(f"找到 3.4.2 BEV 插入点: {j+1}")
                    break
    
    print("\n开始插入引用...")
    
    # 从后往前插入，避免行号偏移
    insertions = []
    
    if targets['3.4.5']:
        insertions.append((targets['3.4.5'], '\n**对应实操**：4.5节VLA决策演示中使用CLIP作为多模态编码器。\n'))
    
    if targets['3.4.3']:
        insertions.append((targets['3.4.3'], '\n**对应实操**：4.3节用LSTM预测Moving MNIST视频的未来10帧。\n'))
    
    if targets['3.4.2_BEV']:
        insertions.append((targets['3.4.2_BEV'], '\n**对应实操**：4.4节展示从真实驾驶图像生成BEV俯视图的完整流程。\n\n'))
    
    if targets['3.2.3']:
        insertions.append((targets['3.2.3'], '\n**对应实操**：4.3节Moving MNIST展示简化的时序预测（视频帧预测）。\n'))
    
    if targets['3.2.2']:
        insertions.append((targets['3.2.2'], '\n**对应实操**：4.2节行为克隆展示最简单的感知-动作映射（无预测）。\n'))
    
    # 按行号倒序排列，从后往前插入
    insertions.sort(key=lambda x: x[0], reverse=True)
    
    for line_num, text in insertions:
        lines.insert(line_num, text)  # line_num是1-based，但insert是0-based，刚好抵消
        print(f"✓ 在行 {line_num+1} 后插入引用")
    
    # 写回文件
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
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
