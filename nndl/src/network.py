"""
network.py
~~~~~~~~~~

a module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation. It is not optimized.
"""

import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """
        :param sizes:列表，sizes中元素的个数表示网络的层数
                          sizes中每个元素表示该对应层神经元的个数
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

    def feedforward(self, a):
        """
        Return the output of the network.
        :param a:input
        :return:
        """
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        使用mini_batch和随机梯度下降训练网络
        :param training_data: 训练数据，列表，每个元素是样本、类标组成的元祖
        :param epochs:
        :param mini_batch_size:
        :param eta:
        :param test_data:如果不为None，则在SGD训练每个epoch之后进行预测
        :return:
        """
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        对于每一个mini batch使用后向传播梯度下降来训练网络的权重：weights biases
        :param mini_batch:元祖（x,y）组成的列表
        :param eta:学习率
        :return:
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            each_sample_nabla_w, each_sample_nabla_b = self.backprop(x, y)
            nabla_w = [nw+enw for nw, enw in zip(nabla_w, each_sample_nabla_w)]
            nabla_b = [nb+enb for nb, enb in zip(nabla_b, each_sample_nabla_b)]
        self.weights = [w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        后向传播
        :param x:单个样本
        :param y:样本对应的类标
        :return:nabla_w, nabla_b, are list of matrix
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # 保存每一层每个神经元激活函数的输出值
        zs = [] # 每一层神经元的输入值
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1]) # 最后一层神经元的输入值关于代价函数的梯度
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # -l表示神经网络的倒数第几层
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_w, nabla_b


    def evaluate(self, test_data):
        """
        返回神经网络预测正确的样本个数
        :param test_data:
        :return:
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for x, y in test_data]
        return sum(int(x==y) for x, y in test_results)


    def cost_derivative(self, output_activation, y):
        """
        求误差函数的梯度
        :param output_activation:
        :param y:
        :return:误差函数的梯度
        """
        return (output_activation - y)

#### Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """
    S函数的导数
    :param z:
    :return:
    """
    return sigmoid(z)*(1-sigmoid(z))