import os
import keras
import numpy as np

from utils.file import dump_to_json

def train_model(data, labels, model_p, save=False, filename=None, save_history=False, h_filename=None):
    model_history = model_p.fit(data, labels, batch_size=10)
    if save and filename:
        model_p.save(filename)
        if save_history and h_filename:
            dump_to_json(model_history.history, h_filename)

if __name__ == '__main__':
    train_data = np.load('train_test/train_data.npy')
    train_labels = np.load('train_test/train_labels.npy')
    print(train_data.shape)
    print(train_labels.shape)
    # exit()
    model_root = '../../../Task2/models'
    history_root = '../../../Task2/data'
    filepath = os.path.join(model_root, 'zhymir_model_2_layer.h5')
    filepath2 = os.path.join(model_root, 'zhymir_model_4_layer.h5')
    filepath3 = os.path.join(model_root, 'zhymir_model_batch_8_4_layer.h5')
    history_filename = os.path.join(history_root, 'zhymir_model_2_layer_history')
    history_filename2 = os.path.join(history_root, 'zhymir_model_4_layer_history')
    history_filename3 = os.path.join(history_root, 'zhymir_model_batch_8_4_layer_history')

    batch_size = 10
    num_classifiers = 16
    model = keras.models.Sequential([
        keras.layers.Dense(units=100, input_shape=(num_classifiers, 10), activation='relu', name='D1'),
        keras.layers.Flatten(),
        keras.layers.Dense(10, name='output_layer', activation='softmax')
    ])
    model.compile('adam', 'categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy(dtype='float64')])
    # train_model(train_data, train_labels, model, True, 'temp', True, 'h_filename ')
    # exit()
    model2 = keras.models.Sequential([
        keras.layers.Dense(units=32,input_shape=(16, 10), name='D1', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(units=100, activation='relu', name='D2'),
        keras.layers.Dense(units=50, activation='relu', name='D3'),
        keras.layers.Dense(10, name='output_layer', activation='softmax')
    ])
    model2.compile('adam', 'categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy(dtype='float64')])
    model3 = keras.models.Sequential([
        keras.layers.Dense(units=32, input_shape=(16, 10), name='D1', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(units=100, activation='relu', name='D2'),
        keras.layers.Dense(units=50, activation='relu', name='D3'),
        keras.layers.Dense(10, name='output_layer', activation='softmax')
    ])
    model3.compile('adam', 'categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy(dtype='float64')])
    history = model.fit(train_data, train_labels, epochs=20, batch_size=10, validation_split=0.1, verbose=0)
    history2 = model2.fit(train_data, train_labels, epochs=20, batch_size=10, validation_split=0.1, verbose=0)
    history3 = model3.fit(train_data, train_labels, epochs=20, batch_size=8, validation_split=0.1, verbose=0)
    model.save(filepath)
    model2.save(filepath2)
    model3.save(filepath3)
    # print(history.history)
    # print(history.history['loss'][0].dtype)
    # print(history.history['accuracy'][0].dtype)
    # dump_to_json(history.history, 'temp')
    # exit()
    dump_to_json(history.history, history_filename)
    dump_to_json(history2.history, history_filename2)
    dump_to_json(history3.history, history_filename3)
