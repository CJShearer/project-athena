# based on ../zhymir_scripts/train_model.py
import os
import keras
import numpy as np

from utils.file import dump_to_json
from utils.file import load_from_json
import matplotlib.pyplot as plt

def train_model(data, labels, model_p, save, filename, save_history, h_filename):
    model_history = model_p.fit(data, labels, batch_size=10)
    if save:
        model_p.save(filename)
        if save_history:
            dump_to_json(model_history.history, h_filename)
if __name__ == '__main__':
    model_config = load_from_json('../../configs/task2/cody_configs/model-mnist.json')
    data_config = load_from_json('../../configs/task2/cody_configs/data-mnist.json')
    WD_config = load_from_json('../../configs/task2/cody_configs/athena-mnist.json')
    # get labels, which are used for each set of 10k predictions
    labels = np.load(os.path.join(data_config.get('dir'), data_config.get('label_file')))
    # get pre-computed predictions of WDs
    predictions = np.load('../../../Task2/data/predictions.npz')['arr_0']

    # we use all predictions on benign samples as training data
    # the keras model will split this into training/validation
    train_data = np.transpose(predictions[0], (1,0,2))

    model_root = '../../../Task2/models'
    history_root = '../../../Task2/data'
    filepath = os.path.join(model_root, 'cody_model.h5')
    history_filename = os.path.join(history_root, 'cody_model_history')
    batch_size = 10
    num_classifiers = 16
    epochs=8
    model = keras.models.Sequential([
        keras.layers.Dense(units=100, input_shape=(num_classifiers, 10), activation='relu', name='D1'),
        keras.layers.Flatten(),
        keras.layers.Dense(10, name='output_layer', activation='softmax')
    ])
    metrics=[keras.metrics.CategoricalAccuracy(dtype='float64')]
    model.compile('adam', 'categorical_crossentropy', metrics=metrics)

    history = model.fit(train_data, labels, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model.save(filepath)
    dump_to_json(history.history, history_filename)
