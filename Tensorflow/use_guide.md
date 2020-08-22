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


