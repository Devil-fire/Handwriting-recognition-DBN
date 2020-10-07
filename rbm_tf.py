import math
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np
import PIL.Image
from util import tile_raster_images


class RBM(object):

    def __init__(self, name, input_size, output_size, opts):
        self._name = name
        self._input_size = input_size
        self._output_size = output_size
        self._opts = opts
        self.init_w = np.zeros([input_size, output_size], np.float32)
        self.init_hb = np.zeros([output_size], np.float32)
        self.init_vb = np.zeros([input_size], np.float32)
        self.w=tf.Variable(tf.zeros([input_size,output_size],dtype=tf.float32))
        self.hb=tf.Variable(tf.zeros([output_size],dtype=tf.float32))
        self.vb=tf.Variable(tf.zeros([input_size],dtype=tf.float32))
        #self.w = np.zeros([input_size, output_size], np.float32)
        #self.hb = np.zeros([output_size], np.float32)
        #self.vb = np.zeros([input_size], np.float32)
        self.k = 1

    def reset_init_parameter(self, init_weights, init_hbias, init_vbias):
        self.init_w = init_weights
        self.init_hb = init_hbias
        self.init_vb = init_vbias

    def propup(self, visible, w, hb):
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    def propdown(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def train(self, X):
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])
        _vw = tf.placeholder("float", [self._input_size, self._output_size])
        _vhb = tf.placeholder("float", [self._output_size])
        _vvb = tf.placeholder("float", [self._input_size])
        _current_vw = np.zeros([self._input_size, self._output_size], np.float32)
        _current_vhb = np.zeros([self._output_size], np.float32)
        _current_vvb = np.zeros([self._input_size], np.float32)
        v0 = tf.placeholder("float", [None, self._input_size])
        h0 = self.sample_prob(self.propup(v0, _w, _hb))
        v1 = self.sample_prob(self.propdown(h0, _w, _vb))
        h1 = self.propup(v1, _w, _hb)
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)
        update_vw = math.exp(-self.k) * self._opts._learning_rate * (positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0])
        update_vvb = math.exp(-self.k) * self._opts._learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_vhb = math.exp(-self.k) * self._opts._learning_rate * tf.reduce_mean(h0 - h1, 0)
        update_w = _w + _vw
        update_vb = _vb + _vvb
        update_hb = _hb + _vhb
        #saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #load_path = saver.restore(sess,self._name+'net.ckpt')
            #print(sess.run(self.hb))
            print("yes")
            old_w = self.init_w
            old_hb = self.init_hb
            old_vb = self.init_vb
            for i in range(self._opts._epoches):
                for start, end in zip(range(0, len(X), self._opts._batchsize),range(self._opts._batchsize,len(X), self._opts._batchsize)):
                    batch = X[start:end]
                    self.k = i*2/self._opts._epoches
                    _current_vw = sess.run(update_vw, feed_dict={
                        v0: batch, _w: old_w, _hb: old_hb, _vb: old_vb,_vw: _current_vw})
                    _current_vhb = sess.run(update_vhb, feed_dict={
                        v0: batch, _w: old_w, _hb: old_hb, _vb: old_vb,_vhb: _current_vhb})
                    _current_vvb = sess.run(update_vvb, feed_dict={
                        v0: batch, _w: old_w, _hb: old_hb, _vb: old_vb,_vvb: _current_vvb})
                    old_w = sess.run(update_w, feed_dict={_w: old_w, _vw: _current_vw})
                    old_hb = sess.run(update_hb, feed_dict={_hb: old_hb, _vhb: _current_vhb})
                    old_vb = sess.run(update_vb, feed_dict={_vb: old_vb, _vvb: _current_vvb})
            self.w = old_w
            self.hb = old_hb
            self.vb = old_vb
            #save_path = saver.save(sess, self._name+'net.ckpt')
            #print ("[+] Model saved in file: %s" % save_path)

    def rbmup(self, X):
        input_X = tf.constant(X)
        out = tf.nn.sigmoid(tf.matmul(input_X, self.w) + self.hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)
