
# load config files: model-config, data-config
import os
import random

import numpy as np

from utils.file import load_from_json
from utils.model import load_lenet, load_pool
from utils.data import subsampling
from utils.data import get_dataloader
from keras.engine.saving import save_model
from keras.callbacks import ModelCheckpoint
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
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callback_list.extend([checkpoint])


model_config = load_from_json('../../configs/task2/zhymir_configs/model-config.json')
data_config = load_from_json('../../configs/task2/zhymir_configs/data-config.json')
WD_config = load_from_json('../../configs/task2/zhymir_configs/task2-athena-mnist.json')

# use load pool to collect WDs and UM
pool, _ = load_pool(trans_configs=WD_config, model_configs=model_config)

test_size = 10

model_file = os.path.join(model_config.get("dir"), model_config.get("um_file"))
target = load_lenet(file=model_file, wrap=True)

data_file = os.path.join(data_config.get('dir'), data_config.get('bs_file'))
data_bs = np.load(data_file)

# load the corresponding true labels
label_file = os.path.join(data_config.get('dir'), data_config.get('label_file'))
labels = np.load(label_file)

# split train test data to use for model training
# bs_data, bs_labels = subsampling(data_bs, labels, 10, 0.001)
# print(bs_data, ' ', bs_labels)
# shuffled_d, shuffled_l = shuffle(bs_data, bs_labels)
bs_data, bs_labels = shuffle(data_bs, labels)

train_data, test_data = bs_data[:test_size], bs_data[-test_size:]
train_labels, test_labels = bs_labels[:test_size], bs_labels[-test_size:]

# optional make dataloader
# batch_size = 1
# train_loader = get_dataloader(train_data, train_labels, batch_size=1, shuffle=True)

# make model trained on WD ouputs, or make model that uses WD outputs as input layer
target.fit(train_data, train_labels)

AE_files = [os.path.join(data_config.get('dir'), AE_name) for AE_name in data_config.get('ae_files')]

for AE_file in AE_files:
    AE_data = np.load(AE_file)
    AE_data, AE_labels = subsampling(AE_data, labels, 10, 0.2)
    AE_train_data, AE_test_data = AE_data[:test_size], AE_data[-test_size:]
    AE_train_labels, AE_test_labels = AE_labels[:test_size], AE_labels[-test_size:]
    for id, model in pool.items():
        if id == 0: # skip the undefended model, which does not transform the image
            continue

        key = 'configs{}'.format(id)
        trans_args = WD_config.get(key)
        print('TRANS CONFIG:', trans_args)
        # transform a small subset
        data_trans = transform(data_bs[:50], trans_args)
        # fit the transformed images by the corresponding model (weak defense)
        target.fit(AE_train_data, AE_train_labels)


filepath = os.path.join('../../../Task2/models', 'zhymir_model.h5')
# optional save model
# save_model(target, filepath=filepath, overwrite=True, include_optimizer=True)
