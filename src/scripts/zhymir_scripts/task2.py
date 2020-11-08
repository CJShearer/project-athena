
# load config files: model-config, data-config
import os
import numpy as np

from utils.file import load_from_json
from utils.model import load_lenet
from utils.data import subsampling
from utils.data import get_dataloader
from keras.engine.saving import save_model

model_config = load_from_json('../../configs/task2/zhymir_configs/model-config.json')
data_config = load_from_json('../../configs/task2/zhymir_configs/data-config.json')
WD_config = load_from_json('../../configs/task2/zhymir_configs/task2-athena-mnist.json')

# use load pool to collect WDs and UM

test_size = 10

model_file = os.path.join(model_config.get("dir"), model_config.get("um_file"))
target = load_lenet(file=model_file, wrap=True)

data_file = os.path.join(data_config.get('dir'), data_config.get('bs_file'))
data_bs = np.load(data_file)

# load the corresponding true labels
label_file = os.path.join(data_config.get('dir'), data_config.get('label_file'))
labels = np.load(label_file)

# split train test data to use for model training
data, labels = subsampling(data_bs, labels, 10, 0.2)
train_data, test_data = data[:test_size], data[-test_size:]
train_labels, test_labels = labels[:test_size], data[-test_size:]

# optional make dataloader
batch_size = 1
train_loader = get_dataloader(train_data, train_labels, batch_size=1, shuffle=True)

# make model trained on WD ouputs, or make model that uses WD outputs as input layer
target.fit(train_data, train_labels)

AE_files = [os.path.join(data_config.get('dir'), AE_name) for AE_name in data_config.get('ae_files')]


filepath = os.path.join('../../../Task2/models', 'zhymir_model.h5')
# optional save model
# save_model(target, filepath=filepath, overwrite=False, include_optimizer=True)
