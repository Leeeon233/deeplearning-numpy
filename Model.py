import numpy as np


class Model(object):
    def __init__(self):
        """
        简单使用列表按顺序存放各层
        """
        self.layers = []

    def add(self, layer):
        """
        向模型中添加一层
        :param layer:  添加的Layer
        """
        self.layers.append(layer)

    def fit(self, X, y, learning_rate, epochs):
        """
        训练
        :param X:                   训练集数据
        :param y:                   训练集标签
        :param learning_rate:       学习率
        :param epochs:              全部数据集学习的轮次
        """
        if self.__loss_function is None:
            raise Exception("compile first")
        # 前馈
        for i in range(epochs):
            loss = 0
            for num in range(len(X)):
                out = X[num]
                for layer in self.layers:
                    out = layer.forward_propagation(out)
                loss += self.__loss_function(out, y[num], True)
                error = self.__loss_function(out, y[num], False)
                for j in range(len(self.layers)):
                    index = len(self.layers) - j - 1
                    error = self.layers[index].back_propagation(error, learning_rate)
            print("epochs {} / {}  loss : {}".format(i + 1, epochs, self.__cal_loss(loss / len(X))))

    def fit_eval(self, X, y, learning_rate, epochs,x_test,y_test):
        """
        训练
        :param X:                   训练集数据
        :param y:                   训练集标签
        :param learning_rate:       学习率
        :param epochs:              全部数据集学习的轮次
        """
        if self.__loss_function is None:
            raise Exception("compile first")
        # 前馈
        for i in range(epochs):
            loss = 0
            for num in range(len(X)):
                out = X[num]
                for layer in self.layers:
                    out = layer.forward_propagation(out)
                loss += self.__loss_function(out, y[num], True)
                error = self.__loss_function(out, y[num], False)
                for j in range(len(self.layers)):
                    index = len(self.layers) - j - 1
                    error = self.layers[index].back_propagation(error, learning_rate)

            res = self.predict(x_test)
            n = 0
            for m, y__ in enumerate(y_test):
                pred = np.argmax(res[m])
                y_ = np.argmax(y__)
                #print("预测", pred, "真实", y_)
                if pred == y_:
                    n += 1
            print("epochs {} / {}  loss : {} acc : {}".format(i + 1, epochs, self.__cal_loss(loss / len(X)),
                                                              n / len(y_test)))

    def compile(self, loss_function):
        """
        编译，目前仅设置损失函数
        :param loss_function:  损失函数的名称
        """
        if loss_function == 'mse':
            self.__loss_function = self.__mse
        if loss_function == 'cross_entropy':
            self.__loss_function = self.__cross_entropy

    def __mse(self, output, y, forward):
        """
        :param output:      预测值
        :param y:           真实值
        :param forward:     是否是前向传播过程
        :return:            loss值
        """
        if forward:
            return 0.5 * ((output - y) ** 2)
        else:
            return output - y

    def __cross_entropy(self, output, y, loss):
        output[output == 0] = 1e-12
        if loss:
            # return -np.mean(y * np.log(output) + (1-y)*np.log(1-output))  # np.nan_to_num
            return -y * np.log(output)
        else:
            # print(y )
            # print(output)
            # print(- y / output)
            return - y / output  # -np.sum(y / output)

    def __cal_loss(self, loss):
        return np.squeeze(np.mean(loss))

    def predict(self, X):
        """
        结果预测
        :param X: 测试集数据
        :return:  对测试集数据的预测
        """
        res = []
        for num in range(len(X)):
            out = X[num]
            for layer in self.layers:
                out = layer.forward_propagation(out)
            res.append(out)
        return np.squeeze(np.array(res))
