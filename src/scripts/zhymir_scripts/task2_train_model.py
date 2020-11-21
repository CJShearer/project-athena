import os
import numpy as np
import keras
from keras.models import load_model

from scripts.zhymir_scripts.task2_functions import *
from utils.file import load_from_json


WD_config = load_from_json('../../configs/task2/cody_configs/athena-mnist.json')
model_config = load_from_json('../../configs/task2/zhymir_configs/model-config.json')
data_config = load_from_json('../../configs/task2/zhymir_configs/data-config.json')
data_file = os.path.join(data_config.get('dir'), data_config.get('bs_file'))
data_bs = np.load(data_file)
print(isinstance(WD_config, dict))
print(data_bs.shape)
print(len(data_bs))
# exit()
# data_bs = np.transpose(data_bs, (1, 0, 2))
label_file = os.path.join(data_config.get('dir'), data_config.get('label_file'))
labels = np.load(label_file)
# print('Shape of labels: ', labels.shape)
# model = keras.models.Sequential([
#     keras.layers.Dense(units=100, input_shape=(16, 10), activation='relu', name='D1'),
#     keras.layers.Flatten(),
#     keras.layers.Dense(10, name='output_layer', activation='softmax')
# ])
# model.compile('adam', 'categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy(dtype='float64')])
model = load_model('new_model.h5', compile=False)
model.compile('adam', 'categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy(dtype='float64')])
new_ensemble = athena_with_model(WD_config, model_config, model)
# print(type(new_ensemble))
# exit()
# new_ensemble.fit(data_bs, labels, batch_size=100)
# new_ensemble.train_adversarial_pgd(data_bs, labels, max_iter=5)
print(new_ensemble.evaluate(data_bs, labels))
attack_args = {'attack': 'pgd', 'description': 'none', 'eps': 0.3, 'eps_step': 0.3/10, 'max_iter': 10,
               'norm': 'l2', 'targeted': False, 'num_random_init': 0, 'random_eps': False}
data_adv = generate(new_ensemble._ensemble, (data_bs[:1000], labels[:1000]), attack_args=attack_args)
print(new_ensemble.evaluate(data_adv, labels[:1000]))
AE_file = ('../../../data/test_AE-mnist-cnn-clean-cw_l2_lr0.01.npy')
AE = np.load(AE_file)
print(new_ensemble.evaluate(AE, labels))
# new_ensemble.save_model('new_model.h5')
# new_ensemble.save_history('new_model')