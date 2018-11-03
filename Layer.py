from abc import abstractmethod
import numpy as np


class Layer(object):

    def __init__(self, **kwargs):
        pass
        # self.__w = 2 * np.random.randn(self.shape[0] + 1, self.shape[1])  # 初始化 融合 bias

    def _activation(self, name, x):
        """
        激活函数
        :param name: 激活函数的名称。
        :param x:    激活函数的自变量。
        :return:     返回激活函数计算得到的值
        """
        if name == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-x))
        elif name == 'tanh':
            return np.tanh(x)
        elif name == 'softmax':
            x = x - np.max(x)  # 防止过大
            exp_x = np.exp(x)
            #print(exp_x)  # TODO
            return exp_x / np.sum(exp_x)
        elif name == 'relu':
            return (np.abs(x) + x) / 2
        elif name == '_relu':
            return (np.abs(0.25 * x) + 0.75 * x) / 2  # 普通relu容易神经元死亡
        elif name == 'none':
            return x
        else:
            raise AttributeError("activation name wrong")

    def _activation_prime(self, name, x):  # 激活函数的求导
        if name == 'sigmoid':
            return self._activation(name, x) * (1 - self._activation(name, x))
        elif name == 'tanh':
            return 1 - np.square(self._activation(name, x))
        elif name == 'softmax':
            x = np.squeeze(x)
            #print(x)
            length = len(x)
            res = np.zeros((length,length))
            # print("length", length)
            for i in range(length):
                for j in range(length):
                    res[i,j] = self.__softmax(i, j, x)
                    #res.append(self.__softmax(i, j, x))
            '''
            print(np.array(res).reshape((length,length)))
            #print("index",np.array(res).reshape((length, length))[index])
            #print("x",x)
            #print("res",np.array(res).reshape((length, length)))
            #print("sum",np.sum(np.array(res).reshape((length, length)), axis=1))
            # return np.array(res).reshape((length, length))[index] # TODO
            return np.array(res).reshape((length, length))[index]
            '''
            return res
        elif name == 'relu':
            return np.where(x > 0, 1, 0)
        elif name == '_relu':
            return np.where(x > 0, 1, 0.5)
        elif name == 'none':
            return 1
        else:
            raise AttributeError("activation name wrong")

    def __softmax(self, i, j, a):
        if i == j:
            return a[i] * (1 - a[i])
        else:
            return -a[i] * a[j]

    @abstractmethod
    def forward_propagation(self, **kwargs):
        pass

    @abstractmethod
    def back_propagation(self, **kwargs):
        pass
