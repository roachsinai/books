import mnist_loader
import network2

training_data, validation_data, test_data = \
    mnist_loader.load_data_wrapper()

"""
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.1, lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=True)
"""

"""
net.SGD(training_data, 30, 10, 1, lmbda=5.0,  0.9575
net.SGD(training_data, 30, 10, 0.5, lmbda=5.0,  0.9597
net.SGD(training_data, 30, 10, 0.3, lmbda=5.0,  0.96
net.SGD(training_data, 30, 10, 0.1, lmbda=5.0,  0.9531
net.SGD(training_data, 30, 10, 0.01, lmbda=5.0,  0.9192


net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost) eta=0.3  .9668
net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost) eta=0.1  .9608

net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost) epoch=30 eta=0.1  .9681
net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost) epoch=30 eta=0.5  .9685
net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost) epoch=60 eta=0.1  .9671 (top1 .9686)
    should early stopping or pick the best performance weights and bias on evaluation data.
net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost) epoch=60 eta=0.5  .9691 (top1 .9712)
    should early stopping or pick the best performance weights and bias on evaluation data.
"""

net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 60, 10, 0.1, lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=True)
