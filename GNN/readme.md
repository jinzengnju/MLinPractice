CNN参数共享
平移不变性：一个小窗口无论移动到图片的哪一个位置，其内部的结构都是一模一样的，因此CNN可以实现参数共享。这就是CNN的精髓所在。

上面讲的图片或者语言，都属于欧式空间的数据，因此才有维度的概念，欧式空间的数据的特点就是结构很规则。
无论卷积核平移到图片中的哪个位置都可以保证其运算结果的一致性，这就是我们所说的「局部平移不变性」.CNN 的卷积本质就是利用这种平移不变性来对扫描的区域进行卷积操作，从而实现了图像特征的提取。




问题2：为什么随机初始化也能有很好的效果
问题3：图卷积神经网络的归一化
问题4：如何理解傅里叶变换


Temporal Interaction Graphs:一个网络，包含着随时间变化的节点interaction连接，希望学习通过时序数据（连续时间/动态）探究边的动态属性（时间依赖性），从而预测网络的相关性。

静态图与动态图
我们所说的deepwalk与node2vec可以非常好的适用与静态图，对于静态图，连接边是没有属性或时间的概念的
实际上，当交互在不同时间或与不同属性关联时，每条边可能具有完全不同的含义，忽略这些影响会导致巨大的信息丢失。具体的，对于统一连接边，实际上这条边可能具有不同的属性，比如是在不同的条件下生成的。
网络的边具有很丰富的交互信息，节点Embedding可同时利用这部分信息。比如“用户”－“股票”之间的交易网络，边上带有丰富的“时间、价格、数量”特征，需要结合这些信息和网络结构，得到“用户”、“股票”的embedding向量，进而用于预测“用户”后续会买哪只股票。




