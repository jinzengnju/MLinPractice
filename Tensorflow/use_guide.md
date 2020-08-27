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

