

import os
import random
import keras
import torch
import numpy as np

from models.athena import Ensemble
from models.keraswrapper import WeakDefense
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
def add_checkpoint(filepath_p, callback_list):
    checkpoint = ModelCheckpoint(filepath_p+'_checkpoint', monitor='loss', verbose=1, save_best_only=False, mode='min')
    # callback_list.extend([checkpoint])
    callback_list = [checkpoint]


def train_model(wd, model, x, y, batch_size_p=1, epochs=1,
                load_from_checkpoint=False, save_checkpoint=False, callback_list_p=None):
    prediction = np.array([wd.predict(x, raw=True)]) # must be in vector because single
    # print(y)
    y = np.array([y])
    # y = keras.utils.to_categorical(y)
    # print(prediction.shape)
    # print(y.shape)
    # print(prediction)
    if load_from_checkpoint and callback_list_p:
        model.fit(prediction, y, callbacks=callback_list_p, epochs=epochs, batch_size=batch_size_p)
    else:
        model.fit(prediction, y, epochs=epochs, batch_size=batch_size_p)
    if save_checkpoint and callback_list_p:
        add_checkpoint(callback_list_p)


def train_ensemble_model(model_p, data_config_p, wd_p, batch_size_p=1, test_size_p=10, save=True, filepath_p=None):
    # Load data
    data_file = os.path.join(data_config_p.get('dir'), data_config_p.get('bs_file'))
    data_bs = np.load(data_file)

    # load the corresponding true labels
    label_file = os.path.join(data_config_p.get('dir'), data_config_p.get('label_file'))
    labels = np.load(label_file)

    # split train test data to use for model training
    # subsampling didn't allow for ratios that were too high, so shuffle data instead
    bs_data, bs_labels = shuffle(data_bs, labels)

    train_data, test_data = bs_data[:test_size_p], bs_data[-test_size_p:]
    train_labels, test_labels = bs_labels[:test_size_p], bs_labels[-test_size_p:]
    # make model trained on WD ouputs, or make model that uses WD outputs as input layer
    # target.fit(train_data, train_labels) [no more]
    call_backs = []
    train_model(wd_p, model_p, train_data, train_labels, batch_size_p=batch_size_p,
                epochs=500, save_checkpoint=True, callback_list_p=call_backs)

    # save_model(target, filepath=filepath, overwrite=False, include_optimizer=True)
    # exit()
    add_checkpoint(filepath_p, call_backs)
    ae_files = [os.path.join(data_config_p.get('dir'), AE_name) for AE_name in data_config_p.get('ae_files')]

    for AE_file in ae_files:
        ae_data = np.load(AE_file)
        ae_data, ae_labels = subsampling(ae_data, labels, 10, 0.2)
        ae_train_data, ae_test_data = ae_data[:test_size_p], ae_data[-test_size_p:]
        ae_train_labels, ae_test_labels = ae_labels[:test_size_p], ae_labels[-test_size_p:]
        train_model(wd_p, model_p, ae_train_data, ae_train_labels, batch_size_p=batch_size_p, epochs=200,
                    load_from_checkpoint=True, save_checkpoint=True, callback_list_p=call_backs)
        # for id, model in pool.items():
        #     if id == 0: # skip the undefended model, which does not transform the image
        #         continue
        #
        #     key = 'configs{}'.format(id)
        #     trans_args = WD_config.get(key)
        #     print('TRANS CONFIG:', trans_args)
        #     # transform a small subset
        #     data_trans = transform(data_bs[:50], trans_args)
        #     fit the transformed images by the corresponding model (weak defense)
        #     target.fit(AE_train_data, AE_train_labels, callbacks=call_backs)
        #     add_checkpoint(filepath, call_backs)

    # print('finished')
    # optional save model
    if save and filepath_p:
        model_p.save(filepath_p)
    return test_data, test_labels


if __name__ == '__main__':
    # load config files: model-config, data-config
    # Change these next 4 lines
    model_config = load_from_json('../../configs/task2/zhymir_configs/model-config.json')
    data_config = load_from_json('../../configs/task2/zhymir_configs/data-config.json')
    WD_config = load_from_json('../../configs/task2/zhymir_configs/task2-athena-mnist.json')
    filepath = os.path.join('../../../Task2/models', 'zhymir_model_batch_size_10.h5')
    # change these 2
    batch_size = 10
    test_size = 10
    # use load pool to collect WDs and UM
    pool, _ = load_pool(trans_configs=WD_config, model_configs=model_config, active_list=True)
    # turns WD into WeakDefense objects for Ensemble
    WD_models = [WeakDefense(pool[WD], WD_config.get('configs'+str(WD))) for WD in pool]
    WDs = Ensemble(WD_models, strategy=None)
    n = len(pool)  # number of WDs in use
    # Create target model for task
    # Change the layers of this model for needs
    target = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(n, 10, 10), name='WD_layer'),
        keras.layers.Flatten(),
        keras.layers.Dense(units=100, activation='relu', name='D1'),
        keras.layers.Dense(10, name='output_layer', activation='softmax')
    ])
    # define loss and optimizer for model
    target.compile('adam', 'categorical_crossentropy')
    train_ensemble_model(target, batch_size_p=batch_size, test_size_p=test_size,
                         data_config_p=data_config, filepath_p=filepath, wd_p=WDs, save=False)

