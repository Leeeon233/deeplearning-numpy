import numpy as np

from Model import Model
from layers.Layers import Dense, Conv, Flatten

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data", one_hot=True)
x_train = np.reshape(mnist.train.images[:800], (800, 28, 28, 1))
y_train = mnist.train.labels[:800]
x_test = np.reshape(mnist.test.images[:1000], (1000, 28, 28, 1))
y_test = mnist.test.labels[:1000]

# print(x_train.shape)

model = Model()
model.add(Conv(16, (7, 7), (28, 28, 1), strides=(2, 2), padding="VALID", activation='tanh'))
model.add(Conv(32, (5, 5), (11, 11, 16), strides=(2, 2), padding="VALID", activation='tanh'))
# model.add(Conv(64, (3, 3), (4, 4, 32), strides=(1, 1), padding="VALID", activation='tanh'))
model.add(Flatten())
model.add(Dense((512, 64), activation='tanh', name='dense'))
model.add(Dense((64, 10), activation='softmax', name='dense2'))
# model.add(Dense((64, 1), activation='none', name='dense2'))
model.compile('cross_entropy')  # cross_entropy
# model.compile('mse')  # cross_entropy
model.fit_eval(x_train, y_train, 0.005, 20, x_test, y_test)
res = model.predict(x_test)
n = 0
for i, y in enumerate(y_test):
    pred = np.argmax(res[i])
    y_ = np.argmax(y)
    # print("预测", pred, "真实", y_)
    if pred == y_:
        n += 1
print("acc", n / len(y_test))
