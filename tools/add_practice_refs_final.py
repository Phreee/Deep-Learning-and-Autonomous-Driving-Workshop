#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在第3章理论部分添加对应第4章实操案例的提及 - 最终版（处理\r\n换行符）
"""

def main():
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print("开始添加实操案例引用...")
    print(f"文件总行数: {len(lines)}")
    
    updated_lines = []
    i = 0
    updates_made = []
    
    while i < len(lines):
        line = lines[i]
        
        # 更新1: 3.2.2 添加行为克隆引用 (line 759)
        if i == 758 and '本课程实操不强求复杂世界模型实现，但要求理解"为什么必须预测"。' in line:
            updated_lines.append(line)
            updated_lines.append('\n')
            updated_lines.append('**对应实操**：4.2节行为克隆展示最简单的感知-动作映射（无预测）。\n')
            updates_made.append('✓ 添加 3.2.2 → 4.2 行为克隆引用 (line 759)')
            i += 1
            continue
            
        # 更新2: 3.2.3 添加世界模型引用 (line 764)
        if i == 763 and '- 李飞飞的世界模型倡议：推动"可生成、可预测"的世界表示' in line:
            updated_lines.append(line)
            updated_lines.append('\n')
            updated_lines.append('**对应实操**：4.3节Moving MNIST展示简化的时序预测（视频帧预测）。\n')
            updates_made.append('✓ 添加 3.2.3 → 4.3 世界模型引用 (line 764)')
            i += 1
            continue
            
        # 更新3: 3.4.3 世界模型时序建模添加引用 (line 869)
        if i == 868 and '世界模型时序建模的目标是预测状态随时间变化的规律。它的重要性在于让系统不仅"看到当前"，还能"预见未来"。本课程的核心实操就是这一点：用前 T 帧预测下一帧。' in line:
            updated_lines.append(line)
            updated_lines.append('\n')
            updated_lines.append('**对应实操**：4.3节用LSTM预测Moving MNIST视频的未来10帧。\n')
            updates_made.append('✓ 添加 3.4.3 → 4.3 时序建模引用 (line 869)')
            i += 1
            continue
            
        # 更新4: 3.4.5 CLIP 添加引用 (line 879)
        if i == 878 and 'CLIP 是视觉-语言对齐的代表模型，让模型学会"图像与文本如何对应"。在智能驾驶中可用于高层语义理解与提示。本课程只做概念理解，不做实现。' in line:
            updated_lines.append(line)
            updated_lines.append('\n')
            updated_lines.append('**对应实操**：4.5节VLA决策演示中使用CLIP作为多模态编码器。\n')
            updates_made.append('✓ 添加 3.4.5 → 4.5 CLIP引用 (line 879)')
            i += 1
            continue
        
        updated_lines.append(line)
        i += 1
    
    # BEV需要单独处理 - 在3.4.2末尾找到"扩展：李飞飞倡导的..."后添加
    final_lines = []
    i = 0
    bev_updated = False
    while i < len(updated_lines):
        line = updated_lines[i]
        final_lines.append(line)
        
        # 查找BEV部分
        if not bev_updated and '扩展：李飞飞倡导的"世界模型"（World Model / World Lab）' in line:
            # 找到这一段的结尾
            while i < len(updated_lines) - 1:
                i += 1
                next_line = updated_lines[i]
                final_lines.append(next_line)
                if '#### 3.4.3' in next_line:
                    # 在3.4.3之前插入
                    final_lines.insert(-1, '\n')
                    final_lines.insert(-1, '**对应实操**：4.4节展示从真实驾驶图像生成BEV俯视图的完整流程。\n')
                    final_lines.insert(-1, '\n')
                    updates_made.append('✓ 添加 3.4.2 → 4.4 BEV引用')
                    bev_updated = True
                    break
        
        i += 1
    
    # 写回文件
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'w', encoding='utf-8') as f:
        f.writelines(final_lines)
    
    print("\n" + "="*60)
    print("✅ 第3章理论与第4章实操关联完成！")
    print("="*60)
    for msg in updates_made:
        print(msg)
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
