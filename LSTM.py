from Layer import Layer
import numpy as np


class LSTMLayer(Layer):
    def __init__(self, units, input_dim):
        super().__init__()
        self.__units = units  # 神经元个数
        self.__input_dim = input_dim  # x维度

        def orthogonal(shape):  # SVD正交初始化
            flat_shape = (shape[0], np.prod(shape[1:]))
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v
            return q.reshape(shape)

        # bias放到w矩阵里
        self.__wf = orthogonal(shape=(self.__units, self.__units + self.__input_dim + self.__units))
        self.__wi = orthogonal(shape=(self.__units, self.__units + self.__input_dim + self.__units))
        self.__wc = orthogonal(shape=(self.__units, self.__units + self.__input_dim + self.__units))
        self.__wo = orthogonal(shape=(self.__units, self.__units + self.__input_dim + self.__units))

    def forward_propagation(self, x_t, ct_1, ht_1):  # 前向传播
        self.__Ct_1 = ct_1
        self.__ht_1 = ht_1
        self.__st = np.row_stack((self.__ht_1, x_t, np.ones((self.__units, 1))))  # 增加一行1 即 bias系数
        self.__ft = self._activation('sigmoid', self.__wf.dot(self.__st))
        self.__it = self._activation('sigmoid', self.__wi.dot(self.__st))
        self.__Ct_hat = self._activation('tanh', self.__wc.dot(self.__st))
        self.__Ct = self.__ft * self.__Ct_1 + self.__it * self.__Ct_hat
        self.__ot = self._activation('sigmoid', self.__wo.dot(self.__st))
        self.__ht = self.__ot * self._activation('tanh', self.__Ct)
        return self.__Ct, self.__ht

    def back_propagationpro(self, Ct_delta, ht_delta, lr, flag):  # 反向传播
        # 反向传播除了最后时刻，其他回传的是St 即 [ht,xt,bias] 更新ht 仅取[: units]
        if flag:
            ht_delta = ht_delta[:self.__units]
        ot_delta = ht_delta * self._activation('tanh', self.__Ct)
        # 未知+=作用
        Ct_delta += ht_delta * self.__ot * self._activation_prime('tanh', self.__Ct) * lr  # TODO  +=
        ft_delta = Ct_delta * self.__Ct_1
        it_delta = Ct_delta * self.__Ct_hat

        Ct_1_delta = Ct_delta * self.__ft
        Ct_hat_delta = Ct_delta * self.__it

        self.__Ct += Ct_delta

        wf_delta = (self.__ft * ft_delta * (1 - self.__ft)).dot(self.__st.T)
        wi_delta = (self.__it * it_delta * (1 - self.__it)).dot(self.__st.T)
        wo_delta = (self.__ot * ot_delta * (1 - self.__ot)).dot(self.__st.T)
        wc_delta = (Ct_hat_delta * (1 - np.square(self.__Ct_hat))).dot(self.__st.T)
        Wh = np.row_stack((self.__wo, self.__wf, self.__wi, self.__wc))
        all_delta = np.row_stack(
            (ot_delta * self.__ot * (1 - self.__ot),
             ft_delta * self.__ft * (1 - self.__ft),
             it_delta * self.__it * (1 - self.__it),
             Ct_hat_delta * (1 - np.square(self.__Ct_hat)))
        )
        ht_1_delta = Wh.T.dot(all_delta)

        self.__wf += wf_delta * lr
        self.__wc += wc_delta * lr
        self.__wi += wi_delta * lr
        self.__wo += wo_delta * lr

        return Ct_1_delta, ht_1_delta
