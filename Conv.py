from Layer import Layer
import numpy as np


class ConvLayer(Layer):
    def __init__(self, filters, kernel_size, input_shape, strides, padding, activation, name="conv"):
        """
        :param filters:         卷积核个数
        :param kernel_size:     卷积核大小
        :param input_shape:     输入shape
        :param strides:         步长
        :param padding:         填充方式
        :param activation:      激活函数名
        :param name:            层名称
        """
        super().__init__()
        self.__filters = filters
        self.__kernel_size = kernel_size
        self.__strides = strides
        self.__padding = padding
        self.activation_name = activation
        self.__input_shape = input_shape  # eg 64*64*3
        self.__input_padding_shape = input_shape
        self.__input = np.zeros(self.__input_shape)
        self.name = name
        self.flag = False

    def _padding_X(self, X):
        """
        对输入进行padding
        :param X:  输入
        :return:   输入padding后的值
        """
        if self.__padding == 'SAME':
            o_w = int(np.ceil(X.shape[0] / self.__strides[0]))
            o_h = int(np.ceil(X.shape[1] / self.__strides[1]))
            self.__output_size = (o_w, o_h, self.__filters)
            p_w = np.max((o_w - 1) * self.__strides[0] + self.__kernel_size[0] - X.shape[0], 0)
            p_h = np.max((o_h - 1) * self.__strides[1] + self.__kernel_size[1] - X.shape[1], 0)
            self.p_l = int(np.floor(p_w / 2))
            self.p_t = int(np.floor(p_h / 2))
            res = np.zeros((X.shape[0] + p_w, X.shape[1] + p_h, X.shape[2]))
            res[self.p_t:self.p_t + X.shape[0], self.p_l:self.p_l + X.shape[1], :] = X
            return res
        elif self.__padding == 'VALID':
            o_w = int(np.ceil((X.shape[0] - self.__kernel_size[0] + 1) / self.__strides[0]))
            o_h = int(np.ceil((X.shape[1] - self.__kernel_size[1] + 1) / self.__strides[1]))
            self.__output_size = (o_w, o_h, self.__filters)
            return X[:self.__strides[0] * (o_w - 1) + self.__kernel_size[0],
                   :self.__strides[1] * (o_h - 1) + self.__kernel_size[1], :]
        else:
            raise ValueError("padding name is wrong")

    def forward_propagation(self, _input):
        """
        前向传播，在前向传播过程中得到输入值，并计算输出shape
        :param _input:  输入值
        :return:        输出值
        """
        self.__input = self._padding_X(_input)
        self.__input_padding_shape = self.__input.shape
        self.__output = np.zeros(self.__output_size)
        #print(self.__output_size)
        if not self.flag:
            self.__kernels = [ConvKernel(self.__kernel_size, self.__input_padding_shape, self.__strides) for _ in
                              range(self.__filters)]  # 由于随机函数，所以不能使用[]*n来创建多个(数值相同)。
            self.flag = True
        for i, kernel in enumerate(self.__kernels):
            self.__output[:, :, i] = kernel.forward_pass(self.__input)
        return self._activation(self.activation_name, self.__output)

    def back_propagation(self, error, lr):
        """
        反向传播过程，对于误差也需要根据padding进行截取或补0
        :param error:   误差
        :param lr:      学习率
        :return:        上一层误差(所有卷积核的误差求平均)
        """
        delta = np.zeros(self.__input_shape)
        for i in range(len(self.__kernels)):
            index = len(self.__kernels) - i - 1
            tmp = self.__kernels[index].back_pass(error[:, :, index], lr, self.activation_name)
            if self.__padding == 'VALID':
                bd = np.ones(self.__input_shape)
                bd[:self.__input_padding_shape[0], :self.__input_padding_shape[1]] = tmp
            elif self.__padding == 'SAME':
                bd = tmp[self.p_t:self.p_t + self.__input_shape[0], self.p_l:self.p_l + self.__input_shape[1]]
            else:
                raise ValueError("padding name is wrong")
            delta += bd

        return delta / len(self.__kernels)


class ConvKernel(Layer):
    """
    这里不需要继承自Layer，但是把激活函数求导过程放在了这里，没改所以还是继承了。
    """

    def __init__(self, kernel_size, input_shape, strides):
        """
        :param kernel_size: 卷积核大小
        :param input_shape: 输入大小
        :param strides:     步长大小
        """
        super().__init__()
        self.__kh = kernel_size[0]
        self.__kw = kernel_size[1]
        self.__input_shape = input_shape
        self.__channel = input_shape[2]
        self.__strides = strides
        # self.__padding = padding
        self.__w = np.random.randn(kernel_size[0], kernel_size[1],
                                   input_shape[2])
        self.__output_shape = (int((input_shape[0] - kernel_size[0]) / strides[0]) + 1,
                               int((input_shape[1] - kernel_size[1]) / strides[1]) + 1)
        self.__input = None
        self.__output = None
        self.__b = np.random.randn(self.__output_shape[0], self.__output_shape[1])

    def __flip_w(self):
        """
        :return: w after flip 180
        """
        return np.fliplr(np.flipud(self.__w))

    def __updata_params(self, w_delta, b_delta, lr):
        self.__w -= w_delta * lr
        self.__b -= b_delta * lr

    def __conv(self, _input, weights, strides, _axis=None):
        """
        卷积运算
        :param _input:      输入
        :param weights:     权重
        :param strides:     步长
        :param _axis:       维度
        :return:
        """
        if _axis is None:  # 矩阵情况
            result = np.zeros((int((_input.shape[0] - weights.shape[0]) / strides[0]) + 1,
                               int((_input.shape[1] - weights.shape[1]) / strides[1]) + 1))
            for h in range(result.shape[0]):
                for w in range(result.shape[1]):
                    result[h, w] = np.sum(_input[h * strides[0]:h * strides[0] + weights.shape[0],
                                          w * strides[1]:w * strides[1] + weights.shape[1]] * weights)
        else:
            result = np.zeros((int((_input.shape[0] - weights.shape[0]) / strides[0]) + 1,
                               int((_input.shape[1] - weights.shape[1]) / strides[1]) + 1,
                               self.__input_shape[2]))
            for h in range(result.shape[0]):
                for w in range(result.shape[1]):
                    result[h, w, :] = np.sum(_input[h * strides[0]:h * strides[0] + weights.shape[0],
                                             w * strides[1]:w * strides[1] + weights.shape[1]] * weights,
                                             axis=_axis)

        return result

    def forward_pass(self, X):
        self.__input = X
        self.__output = self.__conv(X, self.__w, self.__strides) + self.__b
        return self.__output

    def back_pass(self, error, lr, activation_name='none'):
        o_delta = np.zeros((self.__output_shape[0], self.__output_shape[1], self.__channel))
        # 将delta扩展至通道数
        for i in range(self.__channel):
            o_delta[:, :, i] = error
        # 根据输入、步长、卷积核大小计算步长
        X = np.zeros(
            shape=(self.__input_shape[0] + self.__kh - 1, self.__input_shape[1] + self.__kw - 1, self.__channel))

        o_delta_ex = np.zeros(
            (self.__output_shape[0], self.__output_shape[1],
             self.__channel))

        #  根据步长填充0
        for i in range(o_delta.shape[0]):
            for j in range(o_delta.shape[1]):
                X[self.__kh - 1 + i * self.__strides[0],
                self.__kw - 1 + j * self.__strides[1], :] = o_delta[i, j, :]
                # print(o_delta_ex.shape,o_delta.shape)
                o_delta_ex[i, j, :] = o_delta[i, j, :]

        flip_conv_w = self.__conv(X, self.__flip_w(), (1, 1), _axis=(0, 1))
        delta = flip_conv_w * np.reshape(
            self._activation_prime(activation_name, self.__input),
            flip_conv_w.shape)

        w_delta = np.zeros(self.__w.shape)
        for h in range(w_delta.shape[0]):
            for w in range(w_delta.shape[1]):
                if self.__channel == 1:
                    w_delta[h, w, :] = np.sum(self.__input[h:h + o_delta_ex.shape[0],
                                              w:w + o_delta_ex.shape[1]] * o_delta_ex)
                else:
                    w_delta[h, w, :] = np.sum(self.__input[h:h + o_delta_ex.shape[0],
                                              w:w + o_delta_ex.shape[1]] * o_delta_ex, axis=(0, 1))
        self.__updata_params(w_delta, error, lr)
        return delta
