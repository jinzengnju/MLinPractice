# UseGuide


## tf.gather

根据indices从params的指定轴axis索引元素，类似与仅能在指定轴上进行一维索引

返回数据维度：params.shape\[:axis\]+indices.shape+params.shape\[aixs+1:\]

```
data=np.array([[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]],[[5,5,5],[6,6,6]]])
indices=np.array([0,2])
tf.gather(data,indices)

array([[[1, 1, 1],
        [2, 2, 2]],

       [[5, 5, 5],
        [6, 6, 6]]])>
```

* aixs=0，indices为0,2，data为(3,2,3)
* data.shape\[:0\]=()  indices.shape=(2,)  data.shape\[1:\]=(2,3)
* 最后输出维度为(2,2,3)

```
tf.gather(data,indices,axis=2)
array([[[1, 1],
        [2, 2]],

       [[3, 3],
        [4, 4]],

       [[5, 5],
        [6, 6]]])>
```

* data.shape\[:2\]=(3,2)  indices.shape=(2,)  data.shape\[2+1:\]=()
* 输出维度为(3,2,2)

```
params:
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]


 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
  
indices:
[1,0]

gather结果:
[[[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]

 [[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]]
```
* indices 中的每个元素经过索引都变成了 3*4 的张量，所以最终结果就变成了 2*3*4 的张量

```
indices:[[1,0],[0,1]]

tf.gather(params, [[1, 0], [0, 1]]) = [[params[1], params[0]], [params[1], params[0]]] 
[[[[12 13 14 15]
   [16 17 18 19]
   [20 21 22 23]]

  [[ 0  1  2  3]
   [ 4  5  6  7]
   [ 8  9 10 11]]]


 [[[ 0  1  2  3]
   [ 4  5  6  7]
   [ 8  9 10 11]]

  [[12 13 14 15]
   [16 17 18 19]
   [20 21 22 23]]]]

```
* indices 本身是 2*2 的，其中每个元素经过索引都变成了 3*4 的张量，所以最终结果就变成了 2*2*3*4 的张量
* data.shape\[:0\]=()  indices.shape=(2,2)  data.shape\[1:\]=(3,4)

## 模型的保存与加载

### tf.compat.v1.train.import_meta_graph

该函数的主要功能是从.meta计算图文件加载Tensorflow计算图

```
# Create a saver.
saver = tf.compat.v1.train.Saver(...variables...)
# Remember the training_op we want to run by adding it to a collection.
tf.compat.v1.add_to_collection('train_op', train_op)
sess = tf.compat.v1.Session()
for step in xrange(1000000):
    sess.run(train_op)
    if step % 1000 == 0:
        # Saves checkpoint, which by default also exports a meta_graph
        # named 'my-model-global_step.meta'.
        saver.save(sess, 'my-model', global_step=step)


#模型加载      
with tf.Session() as sess:
  new_saver =
  tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
  new_saver.restore(sess, 'my-save-dir/my-model-10000')
  # tf.get_collection() returns a list. In this example we only want
  # the first one.
  train_op = tf.get_collection('train_op')[0]
  for step in xrange(1000000):
    sess.run(train_op)
```
Tensorflow检查点机制是一个非常重要的机制，当保存检查点时，检查点文件一般包含了.index，.meta以及参数值文件，其中的meta定义了计算图结构。

当我们需要加载已经保存的检查点时，可以使用**saver.restore(sess,"./Model/model.ckpt")**，这句话只是恢复参数值，在执行这句话前，应该定义计算图上的所有运算。若不希望重复定义计算图上的运算，可以直接从.meta文件恢复计算图结构。即可以按照如下方式加载一个完整的模型：
```python
saver =tf.train.import_meta_graph("Model/model.ckpt.meta")
saver.restore(sess,"./Model/model.ckpt")
```

tf.train.Saver支持在加载的时候对变量进行重命名，这样可以非常方便的使用滑动平均值。

```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

v = tf.Variable(0, dtype=tf.float32, name="v")
for variables in tf.compat.v1.global_variables():
    print(variables.name)  # v:0

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.compat.v1.global_variables())
for variables in tf.compat.v1.global_variables():
    print(variables.name)  # v:0
    # v/ExponentialMovingAverage:0

saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.assign(v, 10))
    print(sess.run([v, ema.average(v)]))
    sess.run(maintain_averages_op)
    saver.save(sess, "Model/model_ema.ckpt")
    print(sess.run([v, ema.average(v)]))  # [10.0, 0.099999905]

#v:0
#v:0
#v/ExponentialMovingAverage:0
#[10.0, 0.0]
#[10.0, 0.099999905]
```

如代码所示，调用**ema.apply()**之后会生成shadow variable对应的变量。apply方法接受一个var_list作为参数，对var_list中的每个元素都会创建他们对应的shadow variable，并以variable的实际值初始化shadow variable。并且他们会被添加到collection：GraphKeys.MOVING_AVERAGE_VARIABLES。

```python
# Create variables.
var0 = tf.Variable(...)
var1 = tf.Variable(...)
# ... use the variables to build a training model...
...
# Create an op that applies the optimizer.  This is what we usually
# would use as a training op.
opt_op = opt.minimize(my_loss, [var0, var1])

# Create an ExponentialMovingAverage object
ema = tf.train.ExponentialMovingAverage(decay=0.9999)

with tf.control_dependencies([opt_op]):
    # Create the shadow variables, and add ops to maintain moving averages
    # of var0 and var1. This also creates an op that will update the moving
    # averages after each training step.  This is what we will use in place
    # of the usual training op.
    training_op = ema.apply([var0, var1])

...train the model by running training_op...

# Create a Saver that loads variables from their saved shadow values.
shadow_var0_name = ema.average_name(var0)
shadow_var1_name = ema.average_name(var1)
saver = tf.compat.v1.train.Saver({shadow_var0_name: var0, shadow_var1_name:
var1})
saver.restore(...checkpoint filename...)
# var0 and var1 now hold the moving average values
```
- average()与average_name()方法可以让我们非常方便的获取shadow variable以及他们的名字
- 代码中，从checkpoint恢复参数var0与var1时，**我们使用他们对应的滑动平均shadow variable用于恢复（重命名）**，然后导出模型文件用于部署或者评估。

[相关参考](./trainSaver/test.py)

### SavedModel模型保存与检查点保存

我们保存Tensorflow模型的时候，一般会有两种保存方式，上面已经讲了Saver方式保存检查点。接下来讲一下保存savedmodel文件与加载savedmodel文件。

保存saved_model方法：

- 从包含.meta文件的checkpoint恢复模型，并加载参数。保存为savedmodel文件

- 自己重新定义一份模型结构，并从checkpoint恢复参数值。最后保存为savedmodel文件

```python
# -*- coding:UTF-8 -*-
from __future__ import print_function

import tensorflow as tf
import io
import yaml
import argparse

tf.compat.v1.disable_eager_execution()

config_dict={}

def read_config(config_path):
    global config_dict
    f=io.open(config_path,encoding='utf-8')
    config_dict=yaml.load(f)

def saveModel():
  #load and build model
  serialized_tf_example = tf.compat.v1.placeholder(dtype=tf.string,
                                         shape=[None],
                                         name='input_example_tensor')

  #tfrecords协议处理serialized_tf_example得到输入数据data

  #调用写好的model模块创建Model对象

  outputs=tf.reshape(MODEL(data,False),[-1])

  def write_saved_model(path, checkpoint_path, inputs, outputs):
      saver = tf.compat.v1.train.Saver()
      with tf.compat.v1.Session() as sess:
          saver.restore(sess, checkpoint_path)
          export_path = path
          builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)
          tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(inputs)
          tensor_info_y = tf.compat.v1.saved_model.utils.build_tensor_info(outputs)

          prediction_signature = (
              tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                  inputs={'examples': tensor_info_x},
                  outputs={'scores': tensor_info_y},
                  method_name=tf.compat.v1.saved_model.signature_constants
                      .PREDICT_METHOD_NAME))

          builder.add_meta_graph_and_variables(
              sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
              signature_def_map={
                  tf.compat.v1.saved_model.signature_constants
                      .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                      prediction_signature,
              }, )
          builder.save()

  checkpoint = tf.train.get_checkpoint_state(config_dict.get("checkpoint_path"))  
  input_checkpoint = checkpoint.model_checkpoint_path  

  write_saved_model(config_dict.get("export_model_dir"), input_checkpoint, serialized_tf_example, outputs)

```

- 使用savedmodel API保存模型

saved模型加载

- 需要注意的是，当你使用savedmodel API加载模型的时候，一定要开启eager模式。不能将高阶的savedmodel API与低阶的session API混用，否则会出现**无法初始化参数的问题**

```Python
from __future__ import print_function

import tensorflow as tf

import argparse

# tf.compat.v1.disable_eager_execution()

def loadModel():
  MODEL = tf.saved_model.load("path")

  out = MODEL.signatures["serving_default"](tf.constant([example1.SerializeToString(),example2.SerializeToString(),example3.SerializeToString(),example4.SerializeToString(),example5.SerializeToString()]))
  print(out)
```

### 滑动平均
![](../MovingAverage/readme.md)

### collection机制

tensorflow的collection提供了一个全局的存储机制，不会受到变量名生存空间的影响。一处保存，到处可取。
```python
#向collection中存数据
tf.Graph.add_to_collection(name, value)

import tensorflow as tf
tf.reset_default_graph()

w1 = tf.get_variable('w1', shape=[4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
w2 = tf.get_variable('w2', shape=[4], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
tf.add_to_collection('w', w1)
tf.add_to_collection('w', w2)
get_w = tf.get_collection('w')
add_w = tf.add_n(get_w)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print("生成数据的样结果：")
    print(sess.run(w1))
    print(sess.run(w2))
    print("放入集合w数据的结果：")
    print(sess.run(get_w))
    print("集合w中的数据相加的记过")
    print(sess.run(add_w))

#生成数据的样结果：
#[ 0.00524002  0.0679507  -0.0160088  -0.04484038]
#[0.1 0.1 0.1 0.1]
#放入集合w数据的结果：
#[array([ 0.00524002,  0.0679507 , -0.0160088 , -0.04484038], dtype=float32), array([0.1, 0.1, 0.1, 0.1], dtype=float32)]
#集合w中的数据相加的结果：
#[0.10524002 0.16795069 0.0839912  0.05515962]

```
从以上可以看出，在从集合中取变量的时候，直接通过**w**就能将变量w1和w2取出，而我们不用关心这两个变量在定义的时候具体是什么名字。

tf自己也维护一些collection，就像我们定义的所有summary op都会保存在name=tf.GraphKeys.SUMMARIES。这样，tf.get_collection(tf.GraphKeys.SUMMARIES)就会返回所有定义的summary op；tf.Optimizer子类默认优化在tf.GraphKeys.TRAINABLE_VARIABLES下收集的变量，但是也可以传递显式的变量列表。

collection的妙用：计算各个层的正则化损失
```python
import tensorflow as tf
w = tf.get_variable('weight', dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
tf.add_to_collection('regularizer', tf.contrib.layers.l2_regularizer(regular_num=0.001)(w))

shared = tf.nn.conv2d(input, w, [1, stride, stride, 1], padding=padding)

b = tf.get_variable('bias', [out_dim], 'float32', initializer=tf.constant_initializer(0.))
tf.add_to_collection('regularizer', tf.contrib.layers.l2_regularizer(regular_num=0.001)(b))

out = tf.nn.bias_add(shared, b)
regular = tf.add_n(tf.get_collection('regularizer'), 'loss') 
# tf.add_n(inputs,name)
with tf.variable_scope(name='loss') as scope:
    loss = -tf.reduce_sum(label*tf.log(y)) + regular # cross entroy + L2-norm as the loss
```
[参考链接](https://www.cnblogs.com/wanghui-garcia/p/13384518.html)

### Tensorflow加载多个模型

[加载单模型](LoadModel/LoadOneModel.py)

有几个注意点

- 模型输出Tensor被加到“activation”的集合中
- **在定义变量或者运算时，最好对它们进行命名**

    这样是为了方便在加载模型的时候方便的使用指定的一些权重参数，如果不命名的话，这些变量会自动命名为类似“Placeholder_1”的名字
    
[加载多模型](LoadModel/LoadMultiModel.py)

当使用一个session进行加载时，这个会话有自己默认的计算图。如果将所有模型的变量都加载到当前的计算图中，可能会产生冲突。所以当我们使用会话的时候，可以通过**tf.Session(graph=MyGraph)**来指定采用不同的已经创建好的计算图。因此，如果需要加载多个模型，需要将他们加载到不同的图，然后用不同的会话使用它们。

