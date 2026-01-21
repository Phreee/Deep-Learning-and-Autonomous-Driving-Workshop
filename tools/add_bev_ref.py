#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
添加BEV引用
"""

def main():
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"文件总行数: {len(lines)}")
    
    # 找到BEV扩展部分并在后面插入
    for i, line in enumerate(lines):
        if '扩展：李飞飞倡导的"世界模型"' in line and 'BEV/3D 是互补且相辅相成的' in line:
            print(f"找到BEV扩展行: {i+1}")
            print(f"内容: {line[:100]}")
            # 在这行后插入两个新行
            lines.insert(i+1, '\n')
            lines.insert(i+2, '**对应实操**：4.4节展示从真实驾驶图像生成BEV俯视图的完整流程。\n')
            print("✓ 添加 BEV → 4.4 引用")
            break
    
    # 写回文件
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("\n✅ 完成BEV实操引用！")

if __name__ == '__main__':
    main()
