#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理3.3章节的重复内容
"""

def main():
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'rb') as f:
        content = f.read()
    
    text = content.decode('utf-8')
    
    # 删除第一个重复的3.3及其后面混乱的内容
    # 从"### 3.3 智能驾驶技术发展里程碑（历史视角）"开始，到"###**对应实操**"结束
    old_section_1 = '''### 3.3 智能驾驶技术发展里程碑（历史视角）

#### 早期阶段：规则与传感器（1980s-2000s）
- **1980s**：基于规则的车道检测与障碍物识别（ALV项目，CMU NavLab - Ernst Dickmanns, Chuck Thorpe）
- **1995**：CMU NavLab 5 横穿美国（Dean Pomerleau - ALVINN神经网络）
- **2004-2007**：DARPA 挑战赛（Stanley - Sebastian Thrun, 斯坦福；Boss - Chris Urmson, CMU）

#### 深度学习进入视觉感知（2012-2016）
- **2012**：AlexNet（Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton）引爆深度学习
- **2015**：ResNet（Kaiming He 等，MSRA）残差连接解决深层退化
- **2015**：YOLO v1（Joseph Redmon）实时目标检测

###**对应实操**：4.2节行为克隆展示最简单的感知-动作映射（无预测）。

'''
    
    # 删除第二个重复的3.3及其后面混乱的内容
    old_section_2 = '''- 能用 2 句话解释世界状态与世界模型
- 能说清本课程的输入与输出
### 3.3 智能驾驶技术发展里程碑（历史视角）

#### 早期阶段：规则与传感器（1980s-2000s）
- **1980s**：基于规则的车道检测与障碍物识别（ALV项目，CMU NavLab - Ernst Dickmanns, Chuck Thorpe）
- **1995**：CMU NavLab 5 横穿美国（Dean Pomerleau - ALVINN神经网络）
- **2004-2007**：DARPA 挑战赛（Stanley - Sebastian Thrun, 斯坦福；Boss - Chris Urmson, CMU）

#### 深度学习进入视觉感知（2012-2016）
- **2012**：AlexNet（Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton）引爆深度学习
- **2015*#### 3.2.3 世界模型发展脉络（概念级）
- RSSM / Dreamer / PlaNet：用潜变量建模时序状态
- 模型式强化学习：先学世界，再做决策
- 李飞飞的世界模型倡议：推动"可生成、可预测"的世界表示

#### 3.2.4 与实操的对应

本课程第 2 章训练的 CNN/ResNet 可以视为"感知特征抽取器"。  
在世界模型任务中，你会将每一帧的特征作为状态表示，再去预测下一时刻状态。  
也就是说：第 2 章解决"看见什么"，为本章的"预测未来"提供输入基础。

检查点：
'''
    
    new_section_2 = '''- 能用 2 句话解释世界状态与世界模型
- 能说清本课程的输入与输出

'''
    
    # 第三个3.3是正确的，但开头有问题，需要修复
    old_section_3 = '''### 3.3 智能驾驶技术发展里程碑（历史视角）

#### 早期阶段：规则与传感器（1980s-2000s）
- **1980s**：基于规则的车道检测与障碍物识别（ALV项目，CMU NavLab - Ernst Dickmanns, Chuck Thorpe）
- **1995**：CMU NavLab 5 横穿美国（Dean Pomerleau - ALVINN神经网络）
- **2004-2007**：DARPA 挑战赛（Stanley - Sebastian Thrun, 斯坦福；Boss - Chris Urmson, CMU）

#### 深度学习进入视觉感知（2012-2016）
- **2012**：AlexNet（Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton）引爆深度学习
- **2015**：ResNet（Kaiming He 等，MSRA）残差连接解决深层退化
- **2015**：YOLO v1（Joseph Redmon）实时目标检测

###*：ResNet（Kaiming He 等，MSRA）残差连接解决深层退化
- **2015**：YOLO v1（Joseph Redmon）实时目标检测

'''
    
    new_section_3 = '''### 3.3 智能驾驶技术发展里程碑（历史视角）

#### 早期阶段：规则与传感器（1980s-2000s）
- **1980s**：基于规则的车道检测与障碍物识别（ALV项目，CMU NavLab - Ernst Dickmanns, Chuck Thorpe）
- **1995**：CMU NavLab 5 横穿美国（Dean Pomerleau - ALVINN神经网络）
- **2004-2007**：DARPA 挑战赛（Stanley - Sebastian Thrun, 斯坦福；Boss - Chris Urmson, CMU）

#### 深度学习进入视觉感知（2012-2016）
- **2012**：AlexNet（Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton）引爆深度学习
- **2015**：ResNet（Kaiming He 等，MSRA）残差连接解决深层退化
- **2015**：YOLO v1（Joseph Redmon）实时目标检测

'''
    
    print("开始清理重复内容...")
    
    # 执行替换
    if old_section_1 in text:
        text = text.replace(old_section_1, '', 1)
        print("✓ 删除第一个重复的3.3章节")
    else:
        print("✗ 未找到第一个重复")
    
    if old_section_2 in text:
        text = text.replace(old_section_2, new_section_2, 1)
        print("✓ 删除第二个重复的3.3章节")
    else:
        print("✗ 未找到第二个重复")
        
    if old_section_3 in text:
        text = text.replace(old_section_3, new_section_3, 1)
        print("✓ 修复第三个3.3章节的格式")
    else:
        print("✗ 未找到第三个需要修复的部分")
    
    # 写回文件
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'wb') as f:
        f.write(text.encode('utf-8'))
    
    print("\n" + "="*60)
    print("✅ 清理完成！")
    print("="*60)

if __name__ == '__main__':
    main()
