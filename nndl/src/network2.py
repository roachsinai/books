"""network2.py
~~~~~~~~~~~~~~

对network.py进行优化
优化包括：使用交叉熵代价函数、正则化和更好的权重初始化

"""

# standard library
import json
import random
import sys

# third-party library
import numpy as np

# define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """
        :param a: neural network output
        :param y: desired output
        :return: the cost associated with a and y
        """

        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer"""

        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """
        np.nan_to_num is used to ensure numerical stability.
        In particular, if 'a' and 'y' have a 1.0 in the same slot,
        then the expression (1-y)*np.log(1-a) returns nan. The
        np.nan_to_num ensures that that is converted to the correct
        value (0.0).
        :param a: neural network output
        :param y: desired output
        :return: the cost associated with a and y
        """

        return np.sum(np.nan_to_num(-y*np.log(a)
                                    -(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta form the output layer. Note that
        the parameter 'z' is not used by this method. It is included
        in this method's parameters in order to make the interface consistent
        with the delta method for other cost classes.
        """

        return a-y


# main network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        权重、偏置使用'self.default_weight_initializer'进行随机初始化
        :param sizes: 列表，sizes中元素的个数表示网络的层数
                    sizes中每个元素表示该对应层神经元的个数
        :param cost: 代价函数
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """
        使用正太分布对权重向量进行初始化，设第L-1层的神经元个数为n；
        第L层的w初始化的权重服从N(0, 1/n)
        :return: None
        """
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(x, 1) for x in self.sizes[1:]]

    def large_weight_initializer(self):
        """
        使用正太分布对权重向量进行初始化，权重均服从N(0, 1)
        :return: None
        """
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(x, 1) for x in self.sizes[1:]]

    def feedforward(self, a):
        """如果a是输入样本，返回网络的输出"""
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n=0):
        """
        随机梯度下降
        :param training_data: 由(样本，类标)组成的列表
        :param epochs:
        :param mini_batch_size:
        :param eta:
        :param lmbda: regularization parameter lambda
        :param evaluation_data:
        :param monitor_evaluation_cost:
        :param monitor_evaluation_accuracy:
        :param monitor_training_cost:
        :param monitor_training_accuracy:
        :param early_stopping_n:
        :return: 由四个列表组成的元组，每个列表的len等于epochs；
                 即四个类别分别对应每次迭代在evaluation集上的代价、准确率
                 和训练集上的代价和准确率。如果指示标志为False（默认的），返回的
                 四个列表都是None
        """
        n = len(training_data)

        if evaluation_data:
            n_evaluation = len(evaluation_data)

        # early stopping functionality
        best_accuracy = 0
        no_accuracy_change = 0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)

            print("Epoch %s training complete." % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}.".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {}.".format(accuracy / n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}.".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {}.".format(accuracy / n_evaluation))

            # Early stopping:
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                else:
                    no_accuracy_change += 1

                if no_accuracy_change == early_stopping_n:
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        通过小批量样本对权重、偏置进行更新
        :param mini_batch: 元组(x, y)组成的列表
        :param eta: 学习率
        :param lmbda: 正则化参数
        :param n: 训练样本的个数
        :return: None
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            each_sample_nabla_w, each_sample_nabla_b = self.backprop(x, y)
            nabla_w = [nw+enw for nw, enw in zip(nabla_w, each_sample_nabla_w)]
            nabla_b = [nb+enb for nb, enb in zip(nabla_b, each_sample_nabla_b)]
        self.weights = [(1-eta*(lmbda/n))*w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
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
        delta = self.cost.delta(zs[-1], activations[-1], y) * \
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

    def accuracy(self, data, convert=False):
        """
        :param data: 待预测的数据集
        :param convert: 数据是否需要转换的标志；需要这个标志是因为验证集和测试集的类标
                        格式和训练集的格式不同
                        True：预测数据是测试集
        :return: 网络对data预测正确的样本的个数

        之所以训练集的类标有独特的格式（通过将一个类标变成一个向量），是考虑网络学习的效率；

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for x, y in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for x, y in data]

        result_accuracy = sum(int(x == y) for x, y in results)
        return result_accuracy

    def total_cost(self, data, lmbda, convert=False):
        """返回数据集的总代价；
        如果数据集是训练集convert为False，和accuracy函数相反
        有必要的话，可以把`/len(data)`提到外面
        """
        cost = .0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
            cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) # L2正则化代价

        return cost

    def save(self, filename):
        """按照JSON的格式把网络参数保存到`filename`中"""
        data = {"sizes": self.sizes,
                "weights": self.weights,
                "biases": self.biases,
                "cost": str(self.cost.__name__)}
        with open(filename, "w") as f:
            json.dump(data, f)


# loading a network
def load(filename):
    """从filename加载信息，返回Network实例"""
    with open(filename, "r") as f:
        data = json.load(f)

    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


# Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """
    S函数的导数
    :param z:
    :return:
    """
    return sigmoid(z)*(1-sigmoid(z))

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
