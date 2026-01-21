#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
添加BEV引用 - 最终修正版
"""

def main():
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'rb') as f:
        content = f.read()
    
    text = content.decode('utf-8')
    
    # 正确的匹配文本
    old_text = '这与 BEV/3D 是互补且相辅相成的。\r\n\r\n#### 3.4.3 世界模型时序建模'
    new_text = '这与 BEV/3D 是互补且相辅相成的。\r\n\r\n**对应实操**：4.4节展示从真实驾驶图像生成BEV俯视图的完整流程。\r\n\r\n#### 3.4.3 世界模型时序建模'
    
    if old_text in text:
        text = text.replace(old_text, new_text)
        print("✓ 添加 BEV → 4.4 引用")
    else:
        print("✗ 未找到匹配文本")
    
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'wb') as f:
        f.write(text.encode('utf-8'))
    
    print("✅ 完成！")

if __name__ == '__main__':
    main()
