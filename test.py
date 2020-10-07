import input_data
from opts import DLOption
from dbn_tf import DBN
from nn_tf import NN
import numpy as np
import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,mnist.test.labels
opts = DLOption(1, 2, 100)
dbn = DBN([500], opts, trX[0:50000])
start=time.clock()
dbn.train()
opts = DLOption(200, 10, 100)
nn = NN([500], opts, trX[50000:55000], trY[50000:55000],teX,teY)
nn.load_from_dbn(dbn)
nn.train()
end=time.clock()
print(end-start)
print (np.mean(np.argmax(teY, axis=1) == nn.predict(teX)))