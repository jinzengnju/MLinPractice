import numpy as np
import tensorflow as tf
import os


def cnn_model_no_top(features, mode, trainable):
    """
    :param features: 输入
    :param mode: estimator模式
    :param trainable: 该层的变量是否可训练
    :return: 不含最上层全连接层的模型
    """
    input_layer = tf.reshape(features, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, trainable=trainable)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, trainable=trainable)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, shape=[-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, trainable=trainable)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))
    return dropout

def cnn_model_fn(features, labels, mode, params):
    """
    用于构造estimator的model_fn
    :param features: 输入
    :param labels: 标签
    :param mode: 模式
    :param params: 用于训练，迁移学习和微调的dict类型参数
        nb_classes 输入的类别数
    :return: EstimatorSpec
    """
    logits_name = "predictions"
    labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=params["nb_classes"])
    model_no_top = cnn_model_no_top(features["x"], mode, trainable=True)  # mnist是完整的训练
    logits = tf.layers.dense(inputs=model_no_top, units=params["nb_classes"], name=logits_name)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(loss, global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                        predictions=predictions['classes'],
                                        name='accuracy')
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./mnist_model", params={
    "nb_classes": 10
})

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True
)
mnist_classifier.train(input_fn=train_input_fn, steps=2000)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
