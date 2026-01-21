#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
添加3.2.3缺失的实操引用
"""

def main():
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'rb') as f:
        content = f.read()
    
    text = content.decode('utf-8')
    
    old = '李飞飞的世界模型倡议：推动"可生成、可预测"的世界表示\r\n\r\n#### 3.2.4'
    new = '李飞飞的世界模型倡议：推动"可生成、可预测"的世界表示\r\n\r\n**对应实操**：4.3节Moving MNIST展示简化的时序预测（视频帧预测）。\r\n\r\n#### 3.2.4'
    
    if old in text:
        text = text.replace(old, new)
        print("✓ 添加3.2.3的实操引用")
    else:
        print("✗ 未找到匹配内容")
    
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'wb') as f:
        f.write(text.encode('utf-8'))
    
    print("✅ 完成！")

if __name__ == '__main__':
    main()
