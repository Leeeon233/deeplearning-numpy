import numpy as np
from Layer import Layer


class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()
        self.__input_shape = None
        self.activation_name = 'none'

    def forward_propagation(self, _input):
        self.__input_shape = _input.shape
        return _input.flatten()

    def back_propagation(self, error, lr=1):
        #print("flatten shape ",error.shape)
        #print("input shape ",self.__input_shape)
        return np.resize(error, self.__input_shape)
