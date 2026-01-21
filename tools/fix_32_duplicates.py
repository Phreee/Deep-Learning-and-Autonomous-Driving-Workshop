#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理3.2节的重复内容
"""

def main():
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'rb') as f:
        content = f.read()
    
    text = content.decode('utf-8')
    
    # 删除重复的3.2.3, 3.2.4和检查点
    # 保留第一组，删除第二组
    old_duplicate = '''#### 3.2.3 世界模型发展脉络（概念级）
- RSSM / Dreamer / PlaNet：用潜变量建模时序状态
- 模型式强化学习：先学世界，再做决策
- 李飞飞的世界模型倡议：推动"可生成、可预测"的世界表示

**对应实操**：4.3节Moving MNIST展示简化的时序预测（视频帧预测）。

#### 3.2.4 与实操的对应

本课程第 2 章训练的 CNN/ResNet 可以视为"感知特征抽取器"。  
在世界模型任务中，你会将每一帧的特征作为状态表示，再去预测下一时刻状态。  
也就是说：第 2 章解决"看见什么"，为本章的"预测未来"提供输入基础。

检查点：
- 能用 2 句话解释世界状态与世界模型
- 能说清本课程的输入与输出

检查点：
- 能用 2 句话解释世界状态与世界模型
- 能说清本课程的输入与输出
'''
    
    new_text = ''
    
    if old_duplicate in text:
        text = text.replace(old_duplicate, new_text, 1)
        print("✓ 删除重复的3.2.3、3.2.4和检查点")
    else:
        print("✗ 未找到匹配的重复内容")
        # 尝试查找部分内容
        if '#### 3.2.3 世界模型发展脉络' in text:
            count = text.count('#### 3.2.3 世界模型发展脉络')
            print(f"  找到 {count} 个3.2.3标题")
        if '检查点：' in text:
            count = text.count('检查点：')
            print(f"  找到 {count} 个检查点")
    
    # 写回文件
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'wb') as f:
        f.write(text.encode('utf-8'))
    
    print("\n✅ 清理完成！")

if __name__ == '__main__':
    main()
