from keras.datasets import imdb
import numpy as np
from keras import models, layers, optimizers, losses, metrics
import matplotlib.pyplot as plt


# Функция преобразования последовательностей чисел в бинарную матрицу
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Определение модели
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5, batch_size=512)
results = model.evaluate(x_test, y_test)
print(model.predict(x_test)[:10])
print(y_test[:10])


#
# # Создание проверочного набора
# x_val = x_train[:10000]
# partial_x_val = x_train[10000:]
#
# y_val = y_train[:10000]
# partial_y_val = y_train[10000:]
#
#
#
# # Обучение модели
# history = model.fit(
#     partial_x_val,
#     partial_y_val,
#     epochs=20,
#     batch_size=512,
#     validation_data=(x_val, y_val)
# )
#
#
#
# # Формирование графиков потерь
# history_dict = history.history
# print(history_dict.keys())
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
#
# epochs = range(1, 21)
#
# plt.plot(epochs, loss_values, 'bo', label='Trainingloss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()
#
# # Формирование графиков точности
# plt.clf()
# acc_values = history_dict['acc']
# val_acc_values = history_dict['val_acc']
#
# plt.plot(epochs, acc_values, 'bo', label='Training acc')
# plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.show()

