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

