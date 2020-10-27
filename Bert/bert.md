# Bert源码解读

###关于Bert中的mask

Bert在做SelfAttention的时候，假设query（input_ids）为\[batchsize, from_seq\]的Tensor，关于attention序列对象的mask矩阵为\[batchsize, to_seq\]。举一下例子说明：

```python
# input_ids = [
  #     [1, 2, 3, 0, 0],
  #     [1, 3, 5, 6, 1]
  # ]
  # input_mask = [
  #     [1, 1, 1, 0, 0],
  #     [1, 1, 1, 1, 1]
  # ]


  # [
  #     [1, 1, 1, 0, 0],  # 它表示第1个词可以attend to 3个词
  #     [1, 1, 1, 0, 0],  # 它表示第2个词可以attend to 3个词
  #     [1, 1, 1, 0, 0],  # 它表示第3个词可以attend to 3个词
  #     [1, 1, 1, 0, 0],  # 无意义，因为输入第4个词是padding的0
  #     [1, 1, 1, 0, 0]  # 无意义，因为输入第5个词是padding的0
  # ]
  #
  # [
  #     [1, 1, 1, 1, 1],  # 它表示第1个词可以attend to 5个词
  #     [1, 1, 1, 1, 1],  # 它表示第2个词可以attend to 5个词
  #     [1, 1, 1, 1, 1],  # 它表示第3个词可以attend to 5个词
  #     [1, 1, 1, 1, 1],  # 它表示第4个词可以attend to 5个词
  #     [1, 1, 1, 1, 1]  # 它表示第5个词可以attend to 5个词
  # ]

```

因为在输入query中，第一条数据的最后两个是pad的。也许你会有疑问，为什么不是这样的结果？

```python
  # [
  #     [1, 1, 1, 0, 0],  # 它表示第1个词可以attend to 3个词
  #     [1, 1, 1, 0, 0],  # 它表示第2个词可以attend to 3个词
  #     [1, 1, 1, 0, 0],  # 它表示第3个词可以attend to 3个词
  #     [0, 0, 0, 0, 0],  # 无意义，因为输入第4个词是padding的0
  #     [0, 0, 0, 0, 0]  # 无意义，因为输入第5个词是padding的0
  # ]
  #
  # [
  #     [1, 1, 1, 1, 1],  # 它表示第1个词可以attend to 5个词
  #     [1, 1, 1, 1, 1],  # 它表示第2个词可以attend to 5个词
  #     [1, 1, 1, 1, 1],  # 它表示第3个词可以attend to 5个词
  #     [1, 1, 1, 1, 1],  # 它表示第4个词可以attend to 5个词
  #     [1, 1, 1, 1, 1]  # 它表示第5个词可以attend to 5个词
  # ]
```

而Bert将query中第一条数据的pad的0当做普通的id来看待，attention key向量的时候，仍然是\[1, 1, 1, 0, 0\]。那么，岂不是pad为0的位置也会参与计算？

答案不是的。假如经过第一层Transformer Block后得到的Tensor为\[batchsize, seq_len, hidden_size\]。很明显的，如果按照上述过程，那些padding位置得到的向量也不会是0。但这并不会影响后续的计算。因为，在后续的计算中，我们仍然传入的相同的mask Tensor，那些pad位置的向量即使不为0，也不会参与到attention计算。

### 有关attention_heads的设置

```python
      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
          attention_head = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
          #[batch_size * from_seq_length,num_attention_heads * size_per_head]
          attention_heads.append(attention_head)
```

如果你看到这里，attention_heads的设置肯定会让你疑惑。这里这样设置主要是考虑到可能有多种序列的存在，那么将会有多个attention结果。

```python
      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
        #对gid序列进行attention计算
          attention_head_1 = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
              
        #对cid序列进行attention计算
          attention_head_2 = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)              
          #[batch_size * from_seq_length,num_attention_heads * size_per_head]
          attention_heads.append(attention_head_1)
          attention_heads.append(attention_head_2)
          
```

到现在，相信大家也都明白了