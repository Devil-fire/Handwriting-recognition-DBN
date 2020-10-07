from rbm_tf import RBM


class DBN(object):

    def __init__(self, sizes, opts, X):
        self._sizes = sizes
        self._opts = opts
        self._X = X
        self.rbm_list = []
        input_size = X.shape[1]
        for i, size in enumerate(self._sizes):
            self.rbm_list.append(RBM("rbm%d" % i, input_size, size, self._opts))
            input_size = size

    def train(self):
        X = self._X
        for rbm in self.rbm_list:
            rbm.train(X)
            X = rbm.rbmup(X)
