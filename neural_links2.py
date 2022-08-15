import numpy as np


def sigmoid(x):  # Сигмоида
    # Функция активации
    return 1 / (1 + np.exp(-x))


def def_sigmoid(x):  # Производная сигмоиды
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:

    def __init__(self):
        # Веса
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.w7 = np.random.normal()
        self.w8 = np.random.normal()
        self.w9 = np.random.normal()
        self.w10 = np.random.normal()
        self.w11 = np.random.normal()
        self.w12 = np.random.normal()

        # Пороги
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        self.b4 = np.random.normal()

    def feedforward(self, x):

        h1 = sigmoid(self.w1 * x[0] + self.w4 * x[1] + self.w7 * x[2] + self.b1)
        h2 = sigmoid(self.w2 * x[0] + self.w5 * x[1] + self.w8 * x[2] + self.b2)
        h3 = sigmoid(self.w3 * x[0] + self.w6 * x[1] + self.w9 * x[2] + self.b3)
        o1 = sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4)

        return o1

    def train(self, data, all_y_trues):

        learn_rate = 0.1
        epochs = 1000  # Сколько раз пройти по всему набору данных

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w4 * x[1] + self.w7 * x[2] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w2 * x[0] + self.w5 * x[1] + self.w8 * x[2] + self.b2
                h2 = sigmoid(sum_h2)

                sum_h3 = self.w3 * x[0] + self.w6 * x[1] + self.w9 * x[2] + self.b3
                h3 = sigmoid(sum_h3)

                sum_o1 = self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # Считаем частные производные
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Нейрон o1
                d_ypred_d_w10 = h1 * def_sigmoid(sum_o1)
                d_ypred_d_w11 = h2 * def_sigmoid(sum_o1)
                d_ypred_d_w12 = h3 * def_sigmoid(sum_o1)
                d_ypred_d_b4 = def_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w10 * def_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w11 * def_sigmoid(sum_o1)
                d_ypred_d_h3 = self.w12 * def_sigmoid(sum_o1)

                # Нейрон h1
                d_h1_d_w1 = x[0] * def_sigmoid(sum_h1)
                d_h1_d_w4 = x[1] * def_sigmoid(sum_h1)
                d_h1_d_w7 = x[2] * def_sigmoid(sum_h1)
                d_h1_d_b1 = def_sigmoid(sum_h1)

                # Нейрон h2
                d_h2_d_w2 = x[0] * def_sigmoid(sum_h2)
                d_h2_d_w5 = x[1] * def_sigmoid(sum_h2)
                d_h2_d_w8 = x[2] * def_sigmoid(sum_h2)
                d_h2_d_b2 = def_sigmoid(sum_h2)

                # Нейрон h3
                d_h3_d_w3 = x[0] * def_sigmoid(sum_h3)
                d_h3_d_w6 = x[1] * def_sigmoid(sum_h3)
                d_h3_d_w9 = x[2] * def_sigmoid(sum_h3)
                d_h3_d_b3 = def_sigmoid(sum_h3)

                # Обновляем веса и пороги

                # Нейрон h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w4
                self.w7 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w7
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Нейрон h2
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w2
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w5
                self.w8 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w8
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Нейрон h3
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w3
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w6
                self.w9 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w9
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_b3

                # Нейрон o1
                self.w10 -= learn_rate * d_L_d_ypred * d_ypred_d_w10
                self.w11 -= learn_rate * d_L_d_ypred * d_ypred_d_w11
                self.w12 -= learn_rate * d_L_d_ypred * d_ypred_d_w12
                self.b4 -= learn_rate * d_L_d_ypred * d_ypred_d_b4

                # --- Считаем полные потери в конце каждой эпохи
                if epoch % 5 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_loss(all_y_trues, y_preds)
                    print("Epoch %d loss: %.3f" % (epoch, loss))


# Определим набор данных
data = np.array([
  [-2, -1, 0],  # Алиса
  [25, 6, 7],   # Боб
  [17, 4, 1],   # Чарли
  [-15, -6, -8],  # Диана
])
all_y_trues = np.array([
  1,  # Алиса
  0,  # Боб
  0,  # Чарли
  1,  # Диана
])

# Обучаем нашу нейронную сеть!
network = OurNeuralNetwork()
network.train(data, all_y_trues)

# Делаем пару предсказаний
emily = np.array([-7, -3, 5])  # 128 фунтов (52.35 кг), 63 дюйма (160 см)
frank = np.array([20, 2, -2])  # 155 pounds (63.4 кг), 68 inches (173 см)
print("Эмили: %.3f" % network.feedforward(emily).round(0))  # 0.951 - Ж
print("Фрэнк: %.3f" % network.feedforward(frank).round(0))  # 0.039 - М
print("Алиса: %.3f" % network.feedforward(np.array([-2, -1, 1])).round(0))  # 0.951 - Ж

