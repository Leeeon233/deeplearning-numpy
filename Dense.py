from Layer import Layer
import numpy as np


class DenseLayer(Layer):

    def __init__(self, shape, activation, name):
        """
        Dense层初始化。
        :param shape:       如输入神经元有2个，输出神经元有3个。那么shape = (2,3)
        :param activation:  激活函数名称
        :param name:        当前层的名称
        """
        super().__init__()
        self.shape = shape
        self.activation_name = activation
        self.__name = name
        self.__w = 2 * np.random.randn(self.shape[0], self.shape[1])
        self.__b = np.random.randn(1, shape[1])

    def forward_propagation(self, _input):
        """
        Dense层的前向传播实现
        :param _input: 输入的数据，即前一层的输出
        :return:       通过激活函数后的输出
        """
        self.__input = np.array(_input)
        self.__output = self._activation(self.activation_name, self.__input.dot(self.__w) + self.__b)
        return self.__output

    def back_propagation(self, error, learning_rate):
        """
        Dense层的反向传播
        :param error:           后一层传播过来的误差
        :param learning_rate:   学习率
        :param index:
        :return:                传播给前一层的误差
        """
        # print(error.shape)
        o_delta = error * self._activation_prime(self.activation_name, self.__output)
        if self.activation_name == 'softmax':
            o_delta = error.dot(self._activation_prime(self.activation_name, self.__output))
        w_delta = np.matrix(self.__input).T.dot(o_delta)
        input_delta = o_delta.dot(self.__w.T)
        self.__w -= w_delta * learning_rate
        self.__b -= o_delta * learning_rate
        return input_delta

    '''
    def test(self):
    x = [2, 3, 4, ]
    out = self.__activation('softmax', x)
    print(out)
    d1 = -np.sum([0, 1, 0] / out)
    print(d1)
    print((d1 * self.__activation_prime('softmax', out, 1)))
    '''
