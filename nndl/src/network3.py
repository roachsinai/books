"""network3.py
~~~~~~~~~~~~~~

对network2.py进行优化
优化包括：卷积层、池化层
"""

# standard library
import pickle
import gzip

# third-party library
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal.pool import pool_2d

# activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


# Constants
GPU = True
if GPU:
    print("Trying to run under a GPU. If this is not desired, then modify "+\
            "network3.py \n to set the GPU flag to False.")
    try: theano.config.device = 'gpu'
    except: pass
    theano.config.floatX = 'float32'
else:
    print("Trying to run under a CPU. If this is not desired, then modify "+\
            "network3.py \n to set the GPU flag to True.")


# load the mnist data
def load_data_shared(filename="../data/mnist.pkl.gz"):
    with gzip.open(filename, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    def shared(data):
        """将数据放到shared variables，这样可以使用GPU进行操作"""
        shared_x = theano.shared(
                np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
                np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

# main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """
        :param layers: 列表，描述网络的结构；元素为卷积池化层、全连接层、Softmax层等
        :param mini_batch_size: mini_batch样本的个数
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size) # 输入两次x，因为输入层不dropout
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                    prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output # for validation and test
        self.output_dropout = self.layers[-1].output_dropout # for train
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """使用随机梯度下降来进行网络训练"""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data
        # 计算需要执行多少次minibatch
        num_training_batches = int(size(training_data)/mini_batch_size)
        num_validation_batches = int(size(validation_data)/mini_batch_size)
        num_test_batches = int(size(test_data) / mini_batch_size)

        # 定义正则化代价函数，梯度，更新权重
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
                0.5*lmbda*l2_norm_squared / num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                    for param, grad in zip(self.params, grads)]

        # 定义对mini_batch进行梯度下降的函数，以及计算
        # 在每个mini_batch上计算acc的函数
        i = T.lscalar() # mini_batch索引
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        # 进行训练
        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0: # finished on epoch
                    validation_accuracy = np.mean(
                            [validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in range(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

# 定义网络层
class ConvPoolLayer(object):
    """卷积层后面跟一个池化层"""

    def __init__(self, filter_shape, input_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """
        :param filter_shape: 4个元素的元组，（要使用的过滤器的个数，输入的特征图的个数（第一层为1），
                                              输入特征图的高度，宽度）
        :param input_shape: 4个元素的元组，（mini_batch样本的个数，输入的特征图的个数（第一层为1），
                                              输入特征图的高度，宽度）
        :param poolsize: 使用的过滤器矩阵的行数和列数
        """
        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        # 初始化权重和偏置
        # n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize)) 
        self.w = theano.shared(
                np.asarray(
                    # np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                    np.random.normal(
                        loc=0,
                        scale=np.sqrt(1.0/np.linalg.norm(filter_shape[2:])),
                        size=filter_shape),
                    dtype=theano.config.floatX),
                borrow=True)
        self.b = theano.shared(
                np.asarray(
                    np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                    dtype=theano.config.floatX),
                borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.input_shape)
        conv_out = conv2d(
                input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
                input_shape=self.input_shape)
        pooled_out = pool_2d(
                input=conv_out, ws=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
                pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dorpout in the conv layer

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # 初始化权重和偏置
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_in), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "返回该批次样本的准确率"
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # 初始化权重和偏置
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "返回负对数损失"
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "返回该mini_batch的准确率"
        return T.mean(T.eq(y, self.y_out))


# Miscellanea
def size(data):
    "返回数据的样本个数"
    return data[0].get_value(borrow=True).shape[0]

srng = shared_randomstreams.RandomStreams(
    np.random.RandomState(0).randint(999999))
def dropout_layer(layer, p_dropout):
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
