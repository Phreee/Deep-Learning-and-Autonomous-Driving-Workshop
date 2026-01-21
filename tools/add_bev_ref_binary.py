#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
添加BEV引用 - 使用二进制模式保留原始换行符
"""

def main():
    # 以二进制模式读取
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'rb') as f:
        content = f.read()
    
    # 解码为字符串（保留\r\n）
    text = content.decode('utf-8')
    
    # 找到并替换
    old_text = '扩展：李飞飞倡导的"世界模型"（World Model / World Lab）旨在学习通用、可预测的三维世界表示，这与 BEV/3D 是互补且相辅相成的。\r\n\r\n#### 3.4.3'
    new_text = '扩展：李飞飞倡导的"世界模型"（World Model / World Lab）旨在学习通用、可预测的三维世界表示，这与 BEV/3D 是互补且相辅相成的。\r\n\r\n**对应实操**：4.4节展示从真实驾驶图像生成BEV俯视图的完整流程。\r\n\r\n#### 3.4.3'
    
    if old_text in text:
        text = text.replace(old_text, new_text)
        print("✓ 添加 BEV → 4.4 引用")
    else:
        print("✗ 未找到匹配文本")
        # 显示实际文本
        idx = text.find('这与 BEV/3D')
        if idx > 0:
            print(f"实际文本: {repr(text[idx:idx+100])}")
    
    # 以二进制模式写回
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'wb') as f:
        f.write(text.encode('utf-8'))
    
    print("✅ 完成！")

if __name__ == '__main__':
    main()
