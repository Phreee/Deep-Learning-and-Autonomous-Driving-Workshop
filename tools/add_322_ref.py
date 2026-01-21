with open('深度学习及智能驾驶模型实操教程_v2.1.md','rb') as f:
    c=f.read()
t=c.decode('utf-8')

# 精确匹配759-761行之间的内容
old='为什么必须预测"。\r\n\r\n#### 3.2.3 世界模型发展脉络'
new='为什么必须预测"。\r\n\r\n**对应实操**：4.2节行为克隆展示最简单的感知-动作映射（无预测）。\r\n\r\n#### 3.2.3 世界模型发展脉络'

if old in t:
    t=t.replace(old,new,1)
    print("✓ 添加3.2.2的实操引用")
else:
    print("✗ 未找到匹配")
    idx=t.find('为什么必须预测')
    if idx>0:
        print(f"实际文本: {repr(t[idx+16:idx+90])}")

with open('深度学习及智能驾驶模型实操教程_v2.1.md','wb') as f:
    f.write(t.encode('utf-8'))
    
print(f"现在有 {t.count('**对应实操**')} 个实操引用")
