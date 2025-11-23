!pip install d2l==1.0.3 --no-deps  # 先安装 d2l 本身，不自动安装依赖
!pip install numpy pandas matplotlib requests  # 手动安装兼容的依赖（自动选最新兼容版）
import torch
from torch import nn
from d2l import torch as d2l

net =  nn.Sequential( # 使用Sequential容器按顺序堆叠网络层，形成一个完整的神经网络
       # 第一层：卷积层 + 激活函数

    nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),
     # 参数解释：
    # 1：输入通道数（此处为灰度图像，故为1）
    # 6：输出通道数（卷积核数量，提取6种特征）
    # kernel_size=5：卷积核大小为5x5
    # padding=2：在输入图像边缘填充2圈0，使卷积后尺寸不变（保持28x28）
    # Sigmoid激活函数，用于引入非线性变换
    
    # 第二层：平均池化层
    nn.AvgPool2d(kernel_size=2,stride=2),
    # 2D平均池化层
    # 参数解释：
    # kernel_size=2：池化窗口大小为2x2
    # stride=2：步长为2，使输出尺寸减半（28x28 → 14x14）
    # 作用：降低特征图尺寸，减少计算量，增强平移不变性
    
    # 第三层：卷积层 + 激活函数
    nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
    # 参数解释：
    # 6：输入通道数（与上一层输出通道数一致）
    # 16：输出通道数（16种特征）
    # kernel_size=5：卷积核大小5x5（无padding，输出尺寸会减小）
    
    # 每层会创建 16 个卷积核（对应 16 个输出通道）。
# 每个卷积核的尺寸是 5×5×6（高度 × 宽度 × 输入通道数），即每个卷积核会同时作用于输入的所有 6 个通道。

    # 对于 16 个输出通道中的每一个通道，计算步骤如下：
# 取 1 个 5×5×6 的卷积核，分别与输入的 6 个通道（/每个通道是 H×W 的特征图）进行卷积：
# 卷积核的第 1 个 5×5 平面与输入的第 1 个通道卷积，得到 1 个中间特征图。
# 卷积核的第 2 个 5×5 平面与输入的第 2 个通道/卷积，得到第 2 个中间特征图。
# ...（以此类推，直到与第 6 个通道完成卷积）
# 将这 6 个中间特征图逐像素相加，再加上该输出通道对应的偏置（bias），得到该输出通道的最终特征图。

    # 输入14x14 → 输出10x10（14-5+1=10）
    # 再次使用Sigmoid激活函数
    
    # 第四层：平均池化层
    nn.AvgPool2d(kernel_size=2,stride=2),# 平均池
    # 输入10x10 → 输出5x5（尺寸减半）
    
    # 第五层：展平层
    nn.Flatten(),
    # 将多维特征图展平为一维向量
    # 输入为16个5x5的特征图 → 展平后尺寸：16*5*5=400
    
     # 第六层：全连接层 + 激活函数
    nn.Linear(16*5*5,120),nn.Sigmoid(),
     # 全连接层（线性层）
     # 参数解释：
    # 16*5*5=400：输入特征数（与展平后的尺寸一致）
    # 120：输出特征数（将400维特征映射到120维）
    # 激活函数
    
    # 第七层：全连接层 + 激活函数
    nn.Linear(120,84),nn.Sigmoid(),
     # 全连接层
     # 输入120维 → 输出84维
    
     # 第八层：输出层（全连接层）
    nn.Linear(84,10))
    # 全连接层（最终输出层）
    # 输入84维 → 输出10维（对应10个类别，如手写数字0-9）

X = torch .rand(size=(1,1,28,28),dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

def evaluate_accuracy_gpu(net,data_iter,device=None):
  """使用GPU计算模型在数据集上的精度"""
  # 判断模型是否为PyTorch的Module（即PyTorch模型）
  if isinstance(net,nn.Module):
    net.eval()# 将模型切换到评估模式（关闭dropout、批量归一化使用移动平均等）
    if not device: # 如果未指定device，则自动获取模型参数所在的设备
      device = next(iter(net.parameters())).device # 从模型参数中取一个参数，获取其所在设备（如cuda:0）
  #正确预测的数量，总预测的数量
  metric = d2l.Accumulator(2) # 初始化一个累加器（来自d2l库），用于存储两个值：正确预测数、总样本数
  with torch.no_grad():# 禁用梯度计算（评估时不需要反向传播，节省内存和计算资源）
    for X,y in data_iter: # 遍历数据迭代器，获取输入X和标签y
      if isinstance(X,list):# 处理输入X是列表的情况（如BERT等模型的输入可能包含多个部分，如token、segment等）
        #bert微调所需的
        X=[x.to(device) for x in X]# 将列表中的每个元素都转移到指定设备（如GPU）
      else:# 输入X是单个数组/张量的情况（如普通图像数据）
        X = X.to(device)#将输入转移到设备
      y = y.to(device)# 将标签y转移到相同设备（确保与模型和输入在同一设备上
      metric.add(d2l.accuracy(net(X),y),y.numel())# 累加正确预测数和总样本数
  return metric[0]/metric[1]# 总正确数 / 总样本数 = 精度

def train_ch6(net,train_iter,test_iter,num_epochs,lr,device):
  """用GPU训练模型"""
#   定义训练函数train_ch6，参数说明：
# net：待训练的神经网络模型。
# train_iter：训练数据集的迭代器（如DataLoader）。
# test_iter：测试 / 验证数据集的迭代器。
# num_epochs：训练的总轮数。
# lr：学习率（用于优化器）。
# device：训练设备（如torch.device('cuda')或'cpu'）

#初始化模型并配置设备
  def init_weights(m):# 定义内部函数，用于初始化模型权重
    if type(m) == nn.Linear or type(m) == nn.Conv2d: # 仅对全连接层和卷积层初始化
      nn.init.xavier_uniform(m.weight) # 使用Xavier均匀分布初始化权重（避免梯度消失/爆炸）
  net.apply(init_weights)# 对模型的所有子模块递归应用权重初始化函数
  print('===== training device =====',device) # 打印当前训练设备（如cuda:0或cpu）
  net.to(device)# 将模型参数转移到指定设备（GPU/CPU）

#配置优化器和损失函数
  optimizer = torch.optim.SGD(net.parameters(),lr = lr)# 初始化随机梯度下降（SGD）优化器
  # 参数说明：net.parameters()是模型所有可训练参数，lr是学习率
  loss = nn.CrossEntropyLoss() # 定义交叉熵损失函数（适用于分类任务，内置softmax）

#初始化可视化工具和计时器
  animator = d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],
                          legend=['train loss','train acc','test acc'])
   # d2l.Animator是d2l库的可视化工具，用于实时绘制训练曲线：
  # - xlabel：x轴标签为“epoch”
  # - xlim：x轴范围为[1, num_epochs]
  # - legend：曲线标签为训练损失、训练精度、测试精度

  timer,num_batches = d2l.Timer(),len(train_iter)# 初始化计时器和批次数量
   # timer用于统计训练时间，num_batches是每轮训练的总批次数

#训练主循环
  for epoch in range(num_epochs):# 遍历每个训练轮次
    #训练损失之和，训练准确率之和，样本数
    metric = d2l.Accumulator(3)
    net.train() # 将模型切换到训练模式（启用dropout、BN层实时更新等）

    for i,(X,y) in enumerate(train_iter): # 遍历每个批次的输入X和标签y
      timer.start() # 开始记录当前批次的训练时间
      optimizer.zero_grad()# 清零优化器中所有参数的梯度（避免累积上一批次的梯度）
      X,y = X.to(device),y.to(device) # 将输入和标签转移到训练设备（与模型同设备）
      y_hat = net(X) # 模型前向传播，得到预测结果y_hat（形状通常为[batch_size, num_classes]）
      l = loss(y_hat,y)# 计算当前批次的损失（交叉熵损失）
      l.backward()# 损失反向传播，计算所有可训练参数的梯度
      optimizer.step()# 优化器根据梯度更新参数（SGD：w = w - lr * grad）
      with torch.no_grad():# 禁用梯度计算，节省资源
        metric.add(l*X.shape[0],d2l.accuracy(y_hat,y),X.shape[0])
         # 累加：总损失（损失值*批次大小，因为loss返回的是均值）、总正确数、总样本数
      timer.stop()# 停止当前批次计时
      train_l = metric[0]/metric[2] # 计算截至当前的平均训练损失（总损失/总样本数）
      train_acc = metric[1]/metric[2] # 计算截至当前的训练精度（总正确数/总样本数）
    
      #实时可视化训练曲线
      # 每迭代总批次的1/5或最后一个批次时，记录一次指标
      if (i+1)%(num_batches // 5)==0 or i == num_batches - 1:
        animator.add(epoch+(i+1)/num_batches,(train_l,train_acc,None))
         # x轴为“当前轮次+批次比例”（如第1轮的第1/5批次对应x=1.2）
        # y轴记录训练损失和精度（测试精度暂为None
    
    #每轮结束后计算测试精度
    test_acc = evaluate_accuracy_gpu(net,test_iter)# 调用之前定义的函数计算测试精度
    animator.add(epoch+1,(None,None,test_acc)) # 记录本轮的测试精度（x为整数轮次）
  ## 打印最终的损失、训练精度、测试精度（保留3位小数）
  print(f'loss {train_l:.3f},train acc{train_acc:.3f},'
      f'test acc {test_acc:.3f}')

  ## 计算并打印训练速度（总样本数/总时间，单位：样本/秒）
  print(f'{metric[2] * num_epochs / timer.sum():.1f}examples/sec'
      f'on {str(device)}')

lr,num_epochs=0.9,10
train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())

