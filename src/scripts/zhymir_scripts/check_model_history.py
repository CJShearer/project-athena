import numpy as np
import tensorflow as tf
from scripts.zhymir_scripts.train_model import train_model
import keras
from keras.models import load_model
model = load_model('../../../Task2/models/zhymir_model_2_layer.h5')
model2 = load_model('../../../Task2/models/zhymir_model_4_layer.h5')
print(model.weights)
print(model.metrics)
print(model.summary)
# keras.models.Sequential.compile()
# model.compile(optimizer='adam',
#               loss=keras.losses.CategoricalCrossentropy(),
#               metrics=['accuracy'])
result = model.evaluate(np.load('train_test/test_data.npy'), np.load('train_test/test_labels.npy'))
print(result)
exit()
temp = keras.models.Sequential([
    keras.layers.Dense(units=100,input_shape=(16,10), activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
temp.compile('adam', 'categorical_crossentropy')
temp.save('my_model')
new_model = keras.models.load_model('my_model')
print(temp.weights)
print(new_model.weights)