#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理3.2节的重复内容 - 使用\r\n
"""

def main():
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'rb') as f:
        content = f.read()
    
    text = content.decode('utf-8')
    
    # 删除从第二个"#### 3.2.3"到"### 3.3"之前的所有重复内容
    # 查找第二个3.2.3的位置
    first_323 = text.find('#### 3.2.3 世界模型发展脉络')
    if first_323 > 0:
        second_323 = text.find('#### 3.2.3 世界模型发展脉络', first_323 + 10)
        if second_323 > 0:
            # 找到这之后的"### 3.3"
            next_33 = text.find('\r\n### 3.3', second_323)
            if next_33 > 0:
                print(f"删除位置: {second_323} - {next_33}")
                print(f"删除内容: {text[second_323:second_323+100]}...")
                
                # 删除这段重复内容
                text = text[:second_323] + text[next_33+2:]  # +2跳过\r\n
                print("✓ 删除第二个3.2.3到3.3之间的重复内容")
            else:
                print("未找到### 3.3")
        else:
            print("未找到第二个3.2.3")
    else:
        print("未找到第一个3.2.3")
    
    # 写回文件
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'wb') as f:
        f.write(text.encode('utf-8'))
    
    # 验证
    count_323 = text.count('#### 3.2.3')
    count_checkpoint = text.count('检查点：')
    print(f"\n现在有 {count_323} 个3.2.3标题")
    print(f"现在有 {count_checkpoint} 个检查点")
    print("\n✅ 清理完成！")

if __name__ == '__main__':
    main()
