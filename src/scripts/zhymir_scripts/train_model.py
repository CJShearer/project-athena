import keras
import numpy as np

from utils.file import dump_to_json

train_data = np.load('train_test/train_data.npy')
train_labels = np.load('train_test/train_labels.npy')
print(train_data.shape)
print(train_labels.shape)
batch_size = 10
num_classifiers = 16
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(num_classifiers, 10), name='WD_layer'),
    keras.layers.Flatten(),
    keras.layers.Dense(units=100, activation='relu', name='D1'),
    keras.layers.Dense(10, name='output_layer', activation='softmax')
])
model.compile('adam', 'categorical_crossentropy')
model2 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(16, 10), name='WD_layer'),
    keras.layers.Dense(units=32, name='D1', activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(units=100, activation='relu', name='D2'),
    keras.layers.Dense(units=50, activation='relu', name='D3'),
    keras.layers.Dense(10, name='output_layer', activation='softmax')
])
model2.compile('adam', 'categorical_crossentropy')

history = model.fit(train_data, train_labels, batch_size=10, validation_split=0.1)
history2 = model2.fit(train_data, train_labels, batch_size=10, validation_split=0.1)