from keras import layers, models
from keras import optimizers
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu", input_shape=(240, 240, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(14, activation="sigmoid"))

model.summary()

model.compile(loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=1e-4), metrics=["acc"])

types =["normal", "fire", "water", "grass", "electric", "ice", "fighting", "poison", "ground", "flying", "psychic", "bug", "rock", "dragon"]
nb_classes = len(types)

X_train, X_test, y_train, y_test = np.load("./dataset.npy")

X_train = X_train.astype("float") / 255
X_test = X_test.astype("float") / 255

#正解のラベル付け
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = model.fit(  X_train,
                    y_train,
                    epochs=10,
                    batch_size=6,
                    validation_data=(X_test, y_test))

acc = model.history['acc']
val_acc = model.history['val_acc']
loss = model.history['loss']
val_loss = model.history['val_loss']

epochs = range(len(acc))

#学習結果を表示
plt.plot(epochs, acc, 'bo', label='Traning acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Traning and validation accuracy')
plt.legend()
plt.savefig('./acc_graph')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Traning loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Traning and validation loss')
plt.legend()
plt.savefig('./loss')

#モデル・重みの保存
json_string = model.model.to_json()
open('./learned_data/data.json', 'w').write(json_string)

hdf5_file = "./learned_data/data.hdf5"
model.model.save_weights(hdf5_file)

#精度確認
eval_X = np.load("./test_data_X.npy")
eval_Y = np.load("./test_data_Y.npy")

test_Y = np_utils.to_categorical(eval_Y, 14)

score = model.model.evaluate(x=eval_X, y=test_Y)

print('loss=', score[0])
print('accuracy=', score[1])
