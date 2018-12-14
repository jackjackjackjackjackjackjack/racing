import chainer
from chainer import links as L
from chainer import functions as F
from chainer import Variable


class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units*2)  # n_units -> n_units
            self.l3 = L.Linear(None, n_units)
            self.l4 = L.Linear(None, n_out)  # n_units -> n_out
            self.bnorm1 = L.BatchNormalization(n_units)
            self.bnorm2 = L.BatchNormalization(n_units*2)
            self.bnorm3 = L.BatchNormalization(n_units)



    def __call__(self, x, t=None):
#        h = F.dropout(F.relu(self.l1(x)), ratio=0.6)
        h = self.bnorm1(F.relu(self.l1(x)))
#        h = F.dropout(F.relu(self.l2(h)), ratio=0.6)
        h = F.dropout(self.bnorm2(F.relu(self.l2(h))), ratio=0.6)
        h = F.dropout(self.bnorm3(F.relu(self.l3(h))), ratio=0.6)
        h = self.l4(h)
#        loss = F.softmax_cross_entropy(h, t)
#        print(h)
#        print(type(h))
#        print(t[:,self.xp.newaxis])
#        print(type(t[:,self.xp.newaxis]))
        loss = F.mean_squared_error(h, t[:,self.xp.newaxis])
#        accuracy = F.accuracy(h, t)
        accuracy = 1.0
        chainer.report({'loss': loss}, self)
        chainer.report({'accuracy': accuracy}, self)

        if chainer.config.train:
            return loss
        else:
            return h

    def predict(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = self.l4(h)
        return h


class MnistCNN(chainer.Chain):
    def __init__(self, n_out):
        super(MnistCNN, self).__init__()
        with self.init_scope():
            #self.conv1 = ('畳み込み層を定義してね')
            self.conv1 = L.Convolution2D(in_channels=1,out_channels=5,ksize=5,stride=2,pad=2)
            #self.conv2 = ('畳み込み層を定義してね')
            self.conv2 = L.Convolution2D(in_channels=5,out_channels=10,ksize=5,stride=2,pad=2)
            self.l_out = L.Linear(10 * 7 * 7, n_out)

    def __call__(self, x, t):
        #h = 'xのshapeを(len(x), 784)から(len(x), 1, 28, 28)に変形してね'
        h = x.reshape((len(x),1,28,28))
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = self.l_out(h)

        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)
        chainer.report({'loss': loss}, self)
        chainer.report({'accuracy': accuracy}, self)

        if chainer.config.train:
            return loss
        else:
            return h


class CifarCNN(chainer.Chain):
    def __init__(self, n_out):
        super(CifarCNN, self).__init__()
        with self.init_scope():
            self.model = L.VGG16Layers()
            self.l_out = L.Linear(None, n_out)

    def __call__(self, x, t):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            h = self.model(x, layers=['pool5'])['pool5']
        h = self.l_out(h)

        t = self.xp.asarray(t, self.xp.int32)
        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)
        chainer.report({'loss': loss}, self)
        chainer.report({'accuracy': accuracy}, self)

        if chainer.config.train:
            return loss
        else:
            return h

    def predict(self, x):
        h = self.model(x, layers=['pool5'])['pool5']
        h = self.l_out(h)
        predicts = F.argmax(h, axis=1)
        return predicts.data

class RankNet(chainer.Chain):

    def __init__(self, predictor):
        super(RankNet, self).__init__(predictor=predictor)

    def __call__(self, x_i, x_j, t_i, t_j):
        s_i = self.predictor(x_i)
        s_j = self.predictor(x_j)
        s_diff = s_i - s_j
        if t_i.data > t_j.data:
            S_ij = 1
        elif t_i.data < t_j.data:
            S_ij = -1
        else:
            S_ij = 0
        self.loss = (1 - S_ij) * s_diff / 2. + \
            F.math.exponential.Log()(1 + F.math.exponential.Exp()(-s_diff))
        return self.loss
