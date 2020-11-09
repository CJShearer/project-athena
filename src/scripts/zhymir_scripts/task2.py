
# load config files: model-config, data-config
import os
import random
import keras
import torch
import numpy as np

from models.athena import Ensemble
from models.keras import WeakDefense
from models.mnist_cnn import cnn
from utils.file import load_from_json
from utils.model import load_lenet, load_pool
from utils.data import subsampling
from utils.data import get_dataloader
# from keras.engine.saving import save_model
from keras.callbacks import ModelCheckpoint
from keras.models import save_model
from models.image_processor import transform


def shuffle(data, labels):
    # convert to labels
    if len(labels.shape) > 1:
        labels = [np.argmax(p) for p in labels]
    # shuffle the selected ids
    id_s = [i for i in range(data.shape[0])]
    random.shuffle(id_s)
    # get sampled data and labels
    samples = np.asarray([data[i] for i in id_s])
    labels = np.asarray([labels[i] for i in id_s])
    return samples, labels


# modified from toy model on stackoverflow
def add_checkpoint(filepath, callback_list):
    checkpoint = ModelCheckpoint(filepath+'_checkpoint', monitor='loss', verbose=1, save_best_only=False, mode='min')
    # callback_list.extend([checkpoint])
    callback_list = [checkpoint]


def train_model(WD, model, x, y, epochs=1, load_from_checkpoint=False, save_checkpoint=False, callback_list=None):
    prediction = np.array([WD.predict(x, raw=True)]) # must be in vector because single
    # print(y)
    y = np.array([y])
    # y = keras.utils.to_categorical(y)
    # print(prediction.shape)
    # print(y.shape)
    # print(prediction)
    if load_from_checkpoint and callback_list:
        model.fit(prediction, y, callbacks=callback_list, epochs=epochs)
    else:
        model.fit(prediction, y, epochs=epochs)
    if save_checkpoint and callback_list:
        add_checkpoint(callback_list)


if __name__ == '__main__':
    model_config = load_from_json('../../configs/task2/zhymir_configs/model-config.json')
    data_config = load_from_json('../../configs/task2/zhymir_configs/data-config.json')
    WD_config = load_from_json('../../configs/task2/zhymir_configs/task2-athena-mnist.json')
    filepath = os.path.join('../../../Task2/models', 'zhymir_model')
    # use load pool to collect WDs and UM
    pool, _ = load_pool(trans_configs=WD_config, model_configs=model_config, active_list=True)
    # turns WD into WeakDefense objects for Ensemble
    WD_models = [WeakDefense(pool[WD], WD_config.get('configs'+str(WD))) for WD in pool]
    WDs = Ensemble(WD_models, strategy=None)
    n = len(pool)  # number of WDs in use
    # Create target model for task
    target = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(n, 10, 10), name='WD_layer'),
        keras.layers.Flatten(),
        # keras.layers.Input(shape=(10, n)),
        keras.layers.Dense(units=100, activation='relu', name='D1'),
        # keras.layers.Dense(units=100, activation='relu', name='D2'),
        keras.layers.Dense(10, name='output_layer', activation='softmax')
    ])
    target.compile('adam', 'categorical_crossentropy')

    test_size = 10

    # Load data
    data_file = os.path.join(data_config.get('dir'), data_config.get('bs_file'))
    data_bs = np.load(data_file)

    # load the corresponding true labels
    label_file = os.path.join(data_config.get('dir'), data_config.get('label_file'))
    labels = np.load(label_file)

    # split train test data to use for model training
    # subsampling didn't allow for ratios that were too high, so shuffle data instead
    bs_data, bs_labels = shuffle(data_bs, labels)

    train_data, test_data = bs_data[:test_size], bs_data[-test_size:]
    train_labels, test_labels = bs_labels[:test_size], bs_labels[-test_size:]

    # make model trained on WD ouputs, or make model that uses WD outputs as input layer
    # target.fit(train_data, train_labels) [no more]
    call_backs = []
    train_model(WDs, target, train_data, train_labels, epochs=500, save_checkpoint=True, callback_list=call_backs)

    # save_model(target, filepath=filepath, overwrite=False, include_optimizer=True)
    # exit()
    add_checkpoint(filepath, call_backs)
    AE_files = [os.path.join(data_config.get('dir'), AE_name) for AE_name in data_config.get('ae_files')]

    for AE_file in AE_files:
        AE_data = np.load(AE_file)
        AE_data, AE_labels = subsampling(AE_data, labels, 10, 0.2)
        AE_train_data, AE_test_data = AE_data[:test_size], AE_data[-test_size:]
        AE_train_labels, AE_test_labels = AE_labels[:test_size], AE_labels[-test_size:]
        train_model(WDs, target, AE_train_data, AE_train_labels, epochs=200, load_from_checkpoint=True, save_checkpoint=True, callback_list=call_backs)
        # for id, model in pool.items():
        #     if id == 0: # skip the undefended model, which does not transform the image
        #         continue
        #
        #     key = 'configs{}'.format(id)
        #     trans_args = WD_config.get(key)
        #     print('TRANS CONFIG:', trans_args)
        #     # transform a small subset
        #     data_trans = transform(data_bs[:50], trans_args)
            # fit the transformed images by the corresponding model (weak defense)
            # target.fit(AE_train_data, AE_train_labels, callbacks=call_backs)
            # add_checkpoint(filepath, call_backs)

    print('finished')
    # optional save model
    # save_model(target, filepath=filepath, overwrite=False, include_optimizer=True)
    target.save('zhymir_model.h5', '../../../Task2/models')
