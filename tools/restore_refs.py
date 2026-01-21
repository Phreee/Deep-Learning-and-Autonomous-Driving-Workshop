#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新添加丢失的3.2.2和3.2.3的实操引用
"""

def main():
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'rb') as f:
        content = f.read()
    
    text = content.decode('utf-8')
    
    # 添加3.2.2的引用
    old1 = '本课程实操不强求复杂世界模型实现，但要求理解"为什么必须预测"。\r\n\r\n#### 3.2.3'
    new1 = '本课程实操不强求复杂世界模型实现，但要求理解"为什么必须预测"。\r\n\r\n**对应实操**：4.2节行为克隆展示最简单的感知-动作映射（无预测）。\r\n\r\n#### 3.2.3'
    
    if old1 in text:
        text = text.replace(old1, new1)
        print("✓ 添加3.2.2的实操引用")
    else:
        print("✗ 未找到3.2.2匹配内容")
    
    # 添加3.2.3的引用
    old2 = '李飞飞的世界模型倡议：推动"可生成、可预测"的世界表示\r\n\r\n#### 3.2.4'
    new2 = '李飞飞的世界模型倡议：推动"可生成、可预测"的世界表示\r\n\r\n**对应实操**：4.3节Moving MNIST展示简化的时序预测（视频帧预测）。\r\n\r\n#### 3.2.4'
    
    if old2 in text:
        text = text.replace(old2, new2)
        print("✓ 添加3.2.3的实操引用")
    else:
        print("✗ 未找到3.2.3匹配内容")
    
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'wb') as f:
        f.write(text.encode('utf-8'))
    
    # 验证
    count = text.count('**对应实操**')
    print(f"\n现在有 {count} 个实操引用")
    print("✅ 完成！")

if __name__ == '__main__':
    main()
