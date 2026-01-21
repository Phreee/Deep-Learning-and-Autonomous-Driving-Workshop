#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理3.3章节的重复内容 - 使用\r\n换行符
"""

def main():
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'rb') as f:
        content = f.read()
    
    text = content.decode('utf-8')
    
    print("文件字符数:", len(text))
    
    # 查找所有3.3出现的位置
    import re
    pattern = r'### 3\.3 智能驾驶技术发展里程碑'
    matches = list(re.finditer(pattern, text))
    print(f"\n找到 {len(matches)} 个3.3章节标题:")
    for i, m in enumerate(matches):
        print(f"  位置{i+1}: 字符{m.start()}")
    
    if len(matches) != 3:
        print(f"\n警告: 预期3个匹配，实际{len(matches)}个")
        return
    
    # 第一个重复：从第一个3.3到"###**对应实操**"之前
    first_33_start = matches[0].start()
    first_33_end_marker = text.find('###**对应实操**', first_33_start)
    if first_33_end_marker > 0:
        # 找到这个标记后的下一行开始
        first_33_end = text.find('\r\n', first_33_end_marker) + 2
        print(f"\n第一个重复: 字符 {first_33_start} - {first_33_end}")
        print(f"内容预览: {text[first_33_start:first_33_start+100]}...")
        
        # 删除第一个重复
        text = text[:first_33_start] + text[first_33_end:]
        print("✓ 删除第一个重复的3.3章节")
        
        # 重新查找（因为位置变了）
        matches = list(re.finditer(pattern, text))
    
    # 第二个重复：从"- 能说清本课程的输入与输出"后的3.3到下一个"检查点："之前
    if len(matches) >= 2:
        second_33_start = matches[0].start()
        # 向前找到"- 能说清本课程的输入与输出"
        checkpoint_marker = text.rfind('- 能说清本课程的输入与输出\r\n', 0, second_33_start)
        if checkpoint_marker > 0:
            # 从这个检查点之后开始删除
            delete_start = checkpoint_marker + len('- 能说清本课程的输入与输出\r\n')
            
            # 找到第二个3.3的末尾（到下一个"检查点："）
            next_checkpoint = text.find('检查点：', second_33_start)
            if next_checkpoint > 0:
                # 向前找到这一行的开始
                delete_end = text.rfind('\r\n', 0, next_checkpoint)
                
                print(f"\n第二个重复: 字符 {delete_start} - {delete_end}")
                print(f"删除内容预览: {text[delete_start:delete_start+100]}...")
                
                # 删除第二个重复
                text = text[:delete_start] + text[delete_end:]
                print("✓ 删除第二个重复的3.3章节及混乱内容")
                
                # 重新查找
                matches = list(re.finditer(pattern, text))
    
    # 修复第三个3.3的格式错误（###*：ResNet...）
    if len(matches) >= 1:
        third_33_start = matches[0].start()
        # 在这个3.3内查找格式错误
        error_marker = text.find('###*：ResNet', third_33_start)
        if error_marker > 0:
            # 删除从这里到下一行"#### 多传感器融合"之前的重复内容
            next_section = text.find('\r\n\r\n#### 多传感器融合', error_marker)
            if next_section > 0:
                print(f"\n修复格式错误: 字符 {error_marker} - {next_section}")
                text = text[:error_marker] + text[next_section:]
                print("✓ 修复第三个3.3章节的格式错误")
    
    # 写回文件
    with open('深度学习及智能驾驶模型实操教程_v2.1.md', 'wb') as f:
        f.write(text.encode('utf-8'))
    
    # 验证
    final_matches = list(re.finditer(pattern, text))
    print("\n" + "="*60)
    print(f"✅ 清理完成！现在有 {len(final_matches)} 个3.3章节标题")
    print("="*60)

if __name__ == '__main__':
    main()
