from layers.Layer import Layer
import numpy as np


class AveragePooling(Layer):
    def __init__(self, pool_size=(2, 2), strides=(1, 1), channel=3, **kwargs):
        super(AveragePooling).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.__channel = channel

    def forward_propagation(self, _input):
        self.__input_shape = np.shape(_input)
        if self.__channel == 3:
            self.h = (self.__input_shape[0] - self.pool_size[0]) // self.strides[0] + 1
            self.w = (self.__input_shape[1] - self.pool_size[1]) // self.strides[1] + 1
            result = np.zeros((self.h, self.w, self.__input_shape[2]))
            for i in range(self.h):
                for j in range(self.w):
                    for c in range(self.__input_shape[2]):
                        result[i, j, c] = np.mean(_input[
                                                  i * self.strides[0]:i * self.strides[0] + self.pool_size[0],
                                                  i * self.strides[1]:i * self.strides[1] + self.pool_size[1], c])
        elif self.__channel == 2:
            self.h = (self.__input_shape[0] - self.pool_size[0]) // self.strides[0] + 1
            self.w = (self.__input_shape[1] - self.pool_size[1]) // self.strides[1] + 1
            result = np.zeros((self.h, self.w))
            for i in range(self.h):
                for j in range(self.w):
                    result[i, j] = np.mean(_input[
                                           i * self.strides[0]:i * self.strides[0] + self.pool_size[0],
                                           i * self.strides[1]:i * self.strides[1] + self.pool_size[1]])
        else:
            raise ValueError("channels wrong")
        return result

    def back_propagation(self, error, *args):
        result = np.zeros(self.__input_shape)
        if self.__channel == 3:
            for i in range(self.h):
                for j in range(self.w):
                    for c in range(self.__input_shape[2]):
                        result[i * self.strides[0]:i * self.strides[0] + self.pool_size[0],
                        i * self.strides[1]:i * self.strides[1] + self.pool_size[1], c] = \
                            np.ones((self.pool_size[0], self.pool_size[1])) * error[i, j, c]
        elif self.__channel == 2:
            for i in range(self.h):
                for j in range(self.w):
                        result[i * self.strides[0]:i * self.strides[0] + self.pool_size[0],
                        i * self.strides[1]:i * self.strides[1] + self.pool_size[1]] = \
                            np.ones((self.pool_size[0], self.pool_size[1])) * error[i, j]
        return result


if __name__ == '__main__':
    pool = AveragePooling(channel=3)
    # data = np.array([[[[2], [2]], [[2], [1]]], [[[2], [1]], [[2], [1]]]]).resize((2, 2))
    data = np.random.randint(0, 3, (2, 2, 3))
    print(pool.back_propagation(pool.forward_propagation(data)))
