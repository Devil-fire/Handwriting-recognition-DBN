import numpy as np
import math
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

class NN(object):

    def __init__(self, sizes, opts, X, Y,teX,teY):
        self._sizes = sizes
        self._opts = opts
        self._X = X
        self._Y = Y
        self.w_list = []
        self.b_list = []
        self.xx=[]
        self.yy=[]
        self._teX = teX
        self._teY = teY
        self.k = 1
        input_size = X.shape[1]
        for size in self._sizes + [Y.shape[1]]:
            self.w_list.append(np.zeros([input_size, size], np.float32))
            self.b_list.append(np.zeros([size], np.float32))
            input_size = size

    def load_from_dbn(self, dbn):
        assert len(dbn._sizes) == len(self._sizes)
        for i in range(len(self._sizes)):
            assert dbn._sizes[i] == self._sizes[i]
        for i in range(len(self._sizes)):
            self.w_list[i] = dbn.rbm_list[i].w
            self.b_list[i] = dbn.rbm_list[i].hb

    def predict(self, X):
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * len(self.w_list)
        _b = [None] * len(self.b_list)
        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        for i in range(len(self.w_list)):
            _w[i] = tf.constant(self.w_list[i])
            _b[i] = tf.constant(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])
        predict_op = tf.argmax(_a[-1], 1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(predict_op, feed_dict={_a[0]: X})

    def train(self):
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * (len(self._sizes) + 1)
        _b = [None] * (len(self._sizes) + 1)
        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        y = tf.placeholder("float", [None, self._Y.shape[1]])
        for i in range(len(self._sizes) + 1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])
        cost = tf.reduce_mean(tf.square(_a[-1] - y))
        train_op = tf.train.GradientDescentOptimizer(self._opts._learning_rate).minimize(cost)
        predict_op = tf.argmax(_a[-1], 1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self._opts._epoches):
                for start, end in zip(range(0, len(self._X),self._opts._batchsize),range(self._opts._batchsize, len(self._X),self._opts._batchsize)):
                    self.k = i*2/self._opts._epoches
                    sess.run(train_op, feed_dict={_a[0]: self._X[start:end], y: self._Y[start:end]})
                for j in range(len(self._sizes) + 1):
                    self.w_list[j] = sess.run(_w[j])
                    self.b_list[j] = sess.run(_b[j])
                if(i%100==99 and i>1500):
                    print(i + 1)
                    self.yy.append(np.mean(np.argmax(self._teY, axis=1) == self.predict(self._teX)))
                    self.xx.append(i + 1)
                if(i==self._opts._epoches - 1):
                    plt.plot(self.xx,self.yy)
                    plt.savefig("filename.png")
                    #plt.show()
                if(i%1000==0):
                    print(np.mean(np.argmax(self._Y, axis=1) ==sess.run(predict_op, feed_dict={_a[0]: self._X, y: self._Y})),end="")
                    print("test：%.4f"%(np.mean(np.argmax(self._teY, axis=1) == self.predict(self._teX))))

