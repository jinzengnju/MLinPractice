# tfrecords使用

### 相关介绍

- parse_example

对一个批次的Example协议数据进行处理，而parse_single_example则是对单个Example进行处理。One might see performance advantages by batching Example protos with parse_example instead of using this function directly.一般情况下，先做batch再做parse性能上会有较大提升。

```python
import  tensorflow as tf
import numpy as np
def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
  
print(_float_feature(np.exp(1)))

#float_list {
#  value: 2.7182817459106445
#}
#可以使用 .SerializeToString 方法将所有协议消息序列化为二进制字符串：

feature = _float_feature(np.exp(1))
feature.SerializeToString()
#b'\x12\x06\n\x04T\xf8-@'

```

[parse_example用法](https://www.tensorflow.org/api_docs/python/tf/io/parse_example)

- 代码实例

[相关参考](./TFDataSet/read_file.py)

需要注意的是代码用的是eager模式，没有使用session会话模式。

- interleave的作用

首先是可以同时并行的读取多个文件，每个batch包含来自多个文件的数据。这样做的优点是：1.充分打乱数据 2.并行读取多个文件，充分利用多线程