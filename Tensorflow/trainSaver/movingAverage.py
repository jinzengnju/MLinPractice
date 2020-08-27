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
