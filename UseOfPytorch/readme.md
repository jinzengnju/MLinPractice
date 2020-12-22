# Pytorch用法

## 自动求导机制

* requires_grad: 变量冻结、迁移学习、finetune

  * 为变量设置`requires_grad`=false的属性，该设置意味着变量不会被进行梯度更新。**这个标志特别有用，当您想要冻结部分模型时，或者您事先知道不会使用某些参数的梯度。例如，如果要对预先训练的CNN进行优化，只要切换冻结模型中的`requires_grad`标志就足够了**

    ~~~python
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(512, 100)
    
    # Optimize only the classifier
    optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
    ~~~

* volatile: 多用于inference状态

  * volatile属性的设置决定了require_grad is False。对于pytorch下的神经网络，只需要一个`volatile`的输入即可得到一个`volatile`的输出，相对的，需要所有的输入“不需要梯度（requires_grad）”才能得到不需要梯度的输出

    ~~~python
    >>> regular_input = Variable(torch.randn(5, 5))
    >>> volatile_input = Variable(torch.randn(5, 5), volatile=True)
    >>> model = torchvision.models.resnet18(pretrained=True)
    >>> model(regular_input).requires_grad
    True
    >>> model(volatile_input).requires_grad
    False
    >>> model(volatile_input).volatile
    True
    >>> model(volatile_input).creator is None
    True
    ~~~

* pytorch在每次训练迭代都会重新创建图，这就允许使用任意的Python控制流语句，这样可以在每次迭代时改变图的整体形状和大小

## Pytorch API

* 从numpy数组转torch tensor

  ~~~python
  torch.from_numpy(ndarray) → Tensor
  ~~~

* tensor连接：torch.cat

  ~~~
  torch.cat(inputs, dimension=0) → Tensor
  ~~~

* 将模型参数（parameters）以及buffers复制到CPU

  ~~~
  cpu(device_id=None)
  ~~~

* 将模型参数（parameters）以及buffers复制到GPU

  ~~~
  cuda(device_id=None)
  ~~~

* torch.nn.eval()

  将模型设置成`evaluation`模式，仅仅当模型中有`Dropout`和`BatchNorm`是才会有影响。

* load_state_dict(state_dict)

  将`state_dict`中的`parameters`和`buffers`复制到此`module`和它的后代中。`state_dict`中的`key`必须和 `model.state_dict()`返回的`key`一致。 `NOTE`：用来加载模型参数。state_dict (dict) – 保存`parameters`和`persistent buffers`的字典。

* register_buffer(name, tensor)

  给`module`添加一个`persistent buffer`。`persistent buffer`通常被用在这么一种情况：我们需要保存一个状态，但是这个状态不能看作成为模型参数。 例如：, `BatchNorm’s` running_mean 不是一个 `parameter`, 但是它也是需要保存的状态之一。

  ~~~python
  self.register_buffer('running_mean', torch.zeros(num_features))
  self.running_mean
  ~~~

* register_parameter(name, param)

  向`module`添加 `parameter`.`parameter`可以通过注册时候的`name`获取。

* torch.nn.state_dict()

  返回一个字典，保存着`module`的所有状态（`state`）。`parameters`和`persistent buffers`都会包含在字典中，字典的`key`就是`parameter`和`buffer`的 `names`。

  ~~~python
  import torch
  from torch.autograd import Variable
  import torch.nn as nn
  
  class Model(nn.Module):
      def __init__(self):
          super(Model, self).__init__()
          self.conv2 = nn.Linear(1, 2)
          self.vari = Variable(torch.rand([1]))
          self.par = nn.Parameter(torch.rand([1]))
          self.register_buffer("buffer", torch.randn([2,3]))
  
  model = Model()
  print(model.state_dict().keys())
  ~~~

* 多GPU并行

  ~~~
   net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
   output = net(input_var)
  ~~~

  此容器通过将`mini-batch`划分到不同的设备上来实现给定`module`的并行。在`forward`过程中，`module`会在每个设备上都复制一遍，每个副本都会处理部分输入。在`backward`过程中，副本上的梯度会累加到原始`module`上。除了`Tensor`，任何位置参数和关键字参数都可以传到DataParallel中。所有的变量会通过指定的`dim`来划分（默认值为0）。原始类型将会被广播，但是所有的其它类型都会被浅复制。所以如果在模型的`forward`过程中写入的话，将会被损坏。

* class torch.autograd.Variable

  包装一个`Tensor`,并记录用在它身上的`operations`。`Variable`是`Tensor`对象的一个`thin wrapper`，它同时保存着`Variable`的梯度和创建这个`Variable`的`Function`的引用。

## 问题

* 自动求导如何编码历史信息
* Variable上的In-place操作
* torch.storage
* torch.nn.utils.rnn.pack_padded_sequence
* torch.nn.utils.rnn.pad_packed_sequence
* in-place operations