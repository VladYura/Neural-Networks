import numpy as np


def sigmoid(x):  # Сигмоида
    # Функция активации
    return 1 / (1 + np.exp(-x))


def def_sigmoid(x):  # Производная сигмоиды
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class Neuron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class OurNeuralNetwork:

    def __init__(self):
        # Веса
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Пороги
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):

        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)

        return o1

    def train(self, data, all_y_trues):

        learn_rate = 0.1
        epochs = 1000  # Сколько раз пройти по всему набору данных

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # Считаем частные производные
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Нейрон o1
                d_ypred_d_w5 = h1 * def_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * def_sigmoid(sum_o1)
                d_ypred_d_b3 = def_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * def_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * def_sigmoid(sum_o1)

                # Нейрон h1
                d_h1_d_w1 = x[0] * def_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * def_sigmoid(sum_h1)
                d_h1_d_b1 = def_sigmoid(sum_h1)

                # Нейрон h2
                d_h2_d_w3 = x[0] * def_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * def_sigmoid(sum_h2)
                d_h2_d_b2 = def_sigmoid(sum_h2)

                # Обновляем веса и пороги

                # Нейрон h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Нейрон h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Нейрон o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                # --- Считаем полные потери в конце каждой эпохи
                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_loss(all_y_trues, y_preds)
                    print("Epoch %d loss: %.3f" % (epoch, loss))


# Определим набор данных
data = np.array([
  [-2, -1],  # Алиса
  [25, 6],   # Боб
  [17, 4],   # Чарли
  [-15, -6],  # Диана
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
emily = np.array([-7, -3])  # 128 фунтов (52.35 кг), 63 дюйма (160 см)
frank = np.array([20, 2])  # 155 pounds (63.4 кг), 68 inches (173 см)
print("Эмили: %.3f" % network.feedforward(emily).round(1))  # 0.951 - Ж
print("Фрэнк: %.3f" % network.feedforward(frank).round(1))  # 0.039 - М
print("Алиса: %.3f" % network.feedforward(np.array([-2, -1])).round(1))  # 0.951 - Ж


