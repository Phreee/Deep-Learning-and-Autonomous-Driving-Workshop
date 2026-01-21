#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
添加最后两个引用：BEV和CLIP
"""

def main():
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 更新1: BEV部分
    old1 = '扩展：李飞飞倡导的"世界模型"（World Model / World Lab）旨在学习通用、可预测的三维世界表示，这与 BEV/3D 是互补且相辅相成的。\n\n#### 3.4.3'
    new1 = '扩展：李飞飞倡导的"世界模型"（World Model / World Lab）旨在学习通用、可预测的三维世界表示，这与 BEV/3D 是互补且相辅相成的。\n\n**对应实操**：4.4节展示从真实驾驶图像生成BEV俯视图的完整流程。\n\n#### 3.4.3'
    
    if old1 in content:
        content = content.replace(old1, new1, 1)
        print("✓ 添加 3.4.2 BEV → 4.4 引用")
    else:
        print("✗ 未找到 3.4.2 BEV 匹配文本")
        print(f"查找: {repr(old1[:100])}")
    
    # 更新2: CLIP部分  
    old2 = 'CLIP 是视觉-语言对齐的代表模型，让模型学会"图像与文本如何对应"。在智能驾驶中可用于高层语义理解与提示。本课程只做概念理解，不做实现。\n\n#### 3.4.6'
    new2 = 'CLIP 是视觉-语言对齐的代表模型，让模型学会"图像与文本如何对应"。在智能驾驶中可用于高层语义理解与提示。本课程只做概念理解，不做实现。\n\n**对应实操**：4.5节VLA决策演示中使用CLIP作为多模态编码器。\n\n#### 3.4.6'
    
    if old2 in content:
        content = content.replace(old2, new2, 1)
        print("✓ 添加 3.4.5 CLIP → 4.5 引用")
    else:
        print("✗ 未找到 3.4.5 CLIP 匹配文本")
        print(f"查找: {repr(old2[:100])}")
    
    # 写回文件
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n" + "="*60)
    print("✅ 完成最后两个实操引用！")
    print("="*60)

if __name__ == '__main__':
    main()
