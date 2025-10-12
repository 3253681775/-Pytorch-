```python
import torch 
from torch import nn
from d2l import torch as d2l
```


```python
batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
```


```python
#pytorch不会隐式地调整输入的形状
#因此我们在线性层前定义了展平层（flatten)来调整网络输入的形状
net = nn.Sequential(nn.Flatten(),nn.Linear(784,10))
#核心功能：用nn.Sequential（序列容器）串联两个网络层，构建端到端的模型，输入为 28×28 的图像，输出为 10 个类别的原始分数。
#各组件解析：
#nn.Sequential：PyTorch 的 “层容器”，会按传入顺序依次执行各层（前一层的输出作为后一层的输入，无需手动定义前向传播逻辑）；
#nn.Flatten()：图像展平层，作用是将 2D 图像张量（如(batch_size, 1, 28, 28)）转换为 1D 特征向量；
#输入形状（以 Fashion-MNIST 为例）：(batch_size, 通道数, 高度, 宽度) = (256, 1, 28, 28)；
#输出形状：(batch_size, 通道数×高度×宽度) = (256, 1×28×28) = (256, 784)，正好匹配下一层的输入维度；
#nn.Linear(784, 10)：线性层（全连接层），实现 “特征→类别分数” 的线性变换；
#第一个参数784：输入特征维度（展平后的图像特征数）；
#第二个参数10：输出特征维度（Fashion-MNIST 的类别数，对应 10 个类别的原始分数，后续需配合 softmax 转为概率）；
#该层会自动创建两个可训练参数：权重weight（形状(10, 784)）和偏置bias（形状(10,)）。
def init_weights(m):
    #功能：定义一个自定义函数，用于初始化模型中特定层的参数（这里只初始化nn.Linear层的权重）；
    #参数m：函数会被传入模型中的每一个层对象（如nn.Flatten、nn.Linear），通过判断m的类型来决定是否初始化。
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
         #核心逻辑：只对 “线性层（nn.Linear）” 的权重做初始化，其他层（如nn.Flatten无参数，无需初始化）跳过；
        #type(m) == nn.Linear：判断当前传入的层对象m是否为线性层（避免对展平层等无权重的层做无效操作）；
        #nn.init.normal_(m.weight, std=0.01)：用 “正态分布” 初始化线性层的权重m.weight；
        #nn.init.normal_：PyTorch 的参数初始化函数，“_” 表示 “in-place 操作”（直接修改参数值，不返回新对象）；
        #m.weight：线性层的权重参数（形状(10, 784)）；
        #std=0.01：正态分布的标准差为 0.01，均值默认是 0（即权重初始化服从N(0, 0.01²)的正态分布）；
        #为什么这么初始化？：小标准差的正态分布能避免权重过大导致的梯度爆炸，让模型训练更稳定（与之前手动初始化W = torch.normal(0, 0.01)逻辑完全一致）。
        #注意：这里没有初始化偏置m.bias，PyTorch 中nn.Linear的偏置默认初始化为 0（与之前手动初始化b = torch.zeros(10)一致，无需额外操作）。   
net.apply(init_weights);
#功能：将自定义的init_weights函数应用到模型net的所有层，完成参数初始化；
#net.apply(fn)：PyTorch 模型的内置方法，会递归遍历模型中的每一个层对象（包括nn.Sequential容器内的所有层），并对每个层调用fn函数（即init_weights(m)）；
#执行过程：
#首先传入nn.Sequential容器本身（type(m)是nn.Sequential，不满足条件，跳过）；
#接着传入nn.Flatten()层（type(m)是nn.Flatten，无权重，跳过）；
#最后传入nn.Linear(784, 10)层（type(m)是nn.Linear，执行权重初始化，按N(0, 0.01)赋值）；
#末尾的;：在 Jupyter Notebook 中，分号可避免输出函数的返回值（仅执行操作，不显示冗余信息）。
```


```python
#重新审视softmax的实现
loss = nn.CrossEntropyLoss(reduction='none')
#nn.CrossEntropyLoss 是 PyTorch 封装的交叉熵损失函数，本质上是 “softmax 激活 + 交叉熵损失” 的二合一操作，专门用于解决分类问题。它的核心特性是：
#输入要求：直接接收模型最后一层的未规范化预测（logits）（即线性层输出，无需提前做 softmax）；
#自动处理：内部先对输入的 logits 做 softmax 得到概率分布，再计算交叉熵损失；
#数值优化：通过 LogSumExp 技巧避免单独计算 softmax 时的数值溢出问题（如exp(z)过大导致的无穷大）。

#reduction 是 nn.CrossEntropyLoss 的核心参数，用于控制如何处理 “批次中每个样本的损失”，可选值包括：
#'none'：不聚合，返回与输入样本数量相同的损失张量（每个元素对应一个样本的损失）；
#'mean'（默认值）：返回所有样本损失的平均值；
#'sum'：返回所有样本损失的总和。
#这里 reduction='none' 表示保留每个样本的独立损失，不做平均或求和操作
```


```python
#优化算法
trainer = torch.optim.SGD(net.parameters(),lr=0.1)
#optim 是 PyTorch 中用于优化模型参数的模块（torch.optim），包含了多种经典的优化算法（如 SGD、Adam、RMSprop 等），核心作用是根据模型参数的梯度自动更新参数，以最小化损失函数。

#net.parameters() 是 PyTorch 模型的内置方法，用于获取模型中所有需要训练的参数（即requires_grad=True的张量）。在原程序中：
#net 是之前定义的神经网络（如nn.Sequential(nn.Flatten(), nn.Linear(784, 10))）；#
#net.parameters() 会返回该模型中线性层的权重W和偏置b（这两个参数是可训练的，用于拟合数据）。
#例如，对于nn.Linear(784, 10)层，net.parameters()返回的参数包括：
#权重张量W：形状为(10, 784)，requires_grad=True；
#偏置张量b：形状为(10,)，requires_grad=True。

#创建的trainer对象是一个优化器实例，它的核心功能通过以下两个方法实现：
#（1）trainer.zero_grad()：清空梯度
#在每次反向传播前，需要调用此方法清除上一轮计算的梯度（PyTorch 会累积梯度，不清除会导致梯度叠加错误）：
#（2）trainer.step()：执行参数更新
#在计算完损失的梯度（loss.backward()）后，调用此方法根据当前梯度和学习率更新所有参数：
```


```python
num_epochs = 15
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)
```


    
![svg](output_5_0.svg)
    



```python

```
