import tensorflow as tf
### Linear Regression 线性回归###
# Input placeholders
x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')
# Model parameters 定义模型的权值参数
W1 = tf.Variable([0.1], tf.float32)
W2 = tf.Variable([0.1], tf.float32)
W3 = tf.Variable([0.1], tf.float32)
b = tf.Variable([0.1], tf.float32)

# Output 模型的输出
linear_model = tf.identity(W1 * x + W2 * x**2 + W3 * x**3 + b,
                           name='activation_opt')

# Loss 定义损失函数
loss = tf.reduce_sum(tf.square(linear_model - y), name='loss')
# Optimizer and training step 定义优化器运算
optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(loss, name='train_step')

# Remember output operation for later aplication
# Adding it to a collections for easy acces
# This is not required if you NAME your output operation
# 记得将输出操作添加到一个集合中，但如何你命名了输出操作，这一步可以省略
tf.add_to_collection("activation", linear_model)

## Start the session ##
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#  CREATE SAVER
saver = tf.train.Saver()

# Training loop 训练
for i in range(10000):
    sess.run(train, {x: data, y: expected})
    if i % 1000 == 0:
        # You can also save checkpoints using global_step variable
        saver.save(sess, "models/model_name", global_step=i)

# SAVE TensorFlow graph into path models/model_name
# 保存模型到指定路径并命名模型文件名字
saver.save(sess, "models/model_name")