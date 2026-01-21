with open('深度学习及智能驾驶模型实操教程_v2.1.md','rb') as f:
    c=f.read()
t=c.decode('utf-8')

# Pattern 1
old1='为什么必须预测"。\r\n\r\n#### 3.2.3'
new1='为什么必须预测"。\r\n\r\n**对应实操**：4.2节行为克隆展示最简单的感知-动作映射（无预测）。\r\n\r\n#### 3.2.3'
t=t.replace(old1,new1)

# Pattern 2  
old2='世界表示\r\n\r\n#### 3.2.4'
new2='世界表示\r\n\r\n**对应实操**：4.3节Moving MNIST展示简化的时序预测（视频帧预测）。\r\n\r\n#### 3.2.4'
t=t.replace(old2,new2)

with open('深度学习及智能驾驶模型实操教程_v2.1.md','wb') as f:
    f.write(t.encode('utf-8'))
    
print(f"现在有 {t.count('**对应实操**')} 个实操引用")
