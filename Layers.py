from Conv import ConvLayer
from Dense import DenseLayer
from LSTM import LSTMLayer
from Flatten import FlattenLayer


def Conv(filters, kernel_size, input_shape, strides=(1, 1), padding="VALID", activation='none'):
    return ConvLayer(filters, kernel_size, input_shape, strides, padding, activation)


def Dense(shape, activation, name):
    return DenseLayer(shape, activation, name)


def LSTM(units, input_dim):
    return LSTMLayer(units, input_dim)


def Flatten():
    return FlattenLayer()