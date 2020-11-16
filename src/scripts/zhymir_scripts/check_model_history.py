import numpy as np
import tensorflow as tf
from scripts.zhymir_scripts.train_model import train_model
import keras
from keras.models import load_model
model = load_model('../../../Task2/models/zhymir_model_2_layer.h5', compile=False)
model2 = load_model('../../../Task2/models/zhymir_model_4_layer.h5', compile=False)
model3 = load_model('../../../Task2/models/zhymir_model_batch_8_4_layer.h5', compile=False)
model.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=[keras.metrics.CategoricalAccuracy(dtype='float64')])
model2.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=[keras.metrics.CategoricalAccuracy(dtype='float64')])
model3.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=[keras.metrics.CategoricalAccuracy(dtype='float64')])
# print(model.weights)
# print(model.metrics)
print(model.summary())
print(model2.summary())
print(model3.summary())
# keras.models.Sequential.compile()

result = model.evaluate(np.load('../../../Task2/data/train_test/test_data.npy'), np.load(
    '../../../Task2/data/train_test/test_labels.npy'))
print(result)
exit()
temp = keras.models.Sequential([
    keras.layers.Dense(units=100,input_shape=(16, 10), activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
temp.compile('adam', 'categorical_crossentropy')
temp.save('my_model')
new_model = keras.models.load_model('my_model')
print(temp.weights)
print(new_model.weights)