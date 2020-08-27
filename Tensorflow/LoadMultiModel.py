import tensorflow as tf


class ImportGraph():
    """  Importing and running isolated TF graph """

    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            # 从指定路径加载模型到局部图中
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc)
            # There are TWO options how to get activation operation:
            # 两种方式来调用运算或者参数
            # FROM SAVED COLLECTION:
            self.activation = tf.get_collection('activation')[0]
            # BY NAME:
            self.activation = self.graph.get_operation_by_name('activation_opt').outputs[0]

    def run(self, data):
        """ Running the activation operation previously imported """
        # The 'x' corresponds to name of input placeholder
        return self.sess.run(self.activation, feed_dict={"x:0": data})


### Using the class ###
# 测试样例
data = 50  # random data
model = ImportGraph('models/model_name')
result = model.run(data)
print(result)