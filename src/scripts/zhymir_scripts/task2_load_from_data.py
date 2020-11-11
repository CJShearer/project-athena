import os
import keras
import numpy as np

from sklearn.model_selection import train_test_split
from scripts.zhymir_scripts.task2 import make_ensemble, add_checkpoint, train_model
from utils.file import load_from_json

model_config = load_from_json('../../configs/task2/cody_configs/model-mnist.json')
data_config = load_from_json('../../configs/task2/cody_configs/data-mnist.json')
WD_config = load_from_json('../../configs/task2/cody_configs/athena-mnist.json')
filepath = os.path.join('../../../Task2/models', 'zhymir_model_batch_size_10_corrected.h5')
data_path_root = '../../../Task2/data'
labels = np.load(os.path.join(data_config.get('dir'), data_config.get('label_file')))
data = np.load(os.path.join(data_path_root, 'arr_0.npy'))
data = np.transpose(data, (0, 2, 1, 3))
# for i in range(19):
#     print(data[i].shape)
wds = make_ensemble(wd_config=WD_config, model_config=model_config)
batch_size = 10
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(wds._nb_classifiers, 10), name='WD_layer'),
    keras.layers.Flatten(),
    keras.layers.Dense(units=100, activation='relu', name='D1'),
    keras.layers.Dense(10, name='output_layer', activation='softmax')
])
model.compile('adam', 'categorical_crossentropy')
train_x, test_x, train_y, test_y = train_test_split(data[0], labels, test_size=0.2)
total_train_x, total_test_x = np.array([train_x]), np.array([test_x])
total_train_y, total_test_y = np.array([train_y]), np.array([test_y])
model.fit(train_x, train_y, epochs=10, batch_size=10, validation_split=0.1)
call_back = []
add_checkpoint(filepath_p=filepath, callback_list=call_back)
for idx in range(len(data_config.get('ae_files'))):
    train_x, test_x, train_y, test_y = train_test_split(data[idx], labels, test_size=0.2)
    total_train_x = np.append(total_train_x, np.array([train_x]))
    total_test_x = np.append(total_test_x, np.array([test_x]))
    total_train_y = np.append(total_train_y, np.array([train_y]))
    total_test_y = np.append(total_test_y, np.array([test_y]))
    model.fit(train_x, train_y,  callbacks=call_back, epochs=10, batch_size=batch_size)
    add_checkpoint(filepath_p=filepath, callback_list=call_back)
np.savez_compressed('train_test', train_data=total_train_x, test_data=total_test_x,
                    train_labels=total_train_y, test_labels=total_test_y)
model.save(filepath)
