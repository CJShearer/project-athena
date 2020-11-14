import keras
import numpy as np


train_data = np.load('')
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(16, 10), name='WD_layer'),
    keras.layers.Dense(units=32, name='D1', activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(units=100, activation='relu', name='D2'),
    keras.layers.Dense(units=50, activation='relu', name='D3'),
    keras.layers.Dense(10, name='output_layer', activation='softmax')
])
model.compile('adam', 'categorical_crossentropy')
