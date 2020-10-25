import os
import numpy as np
from tutorials.craft_adversarial_examples import generate_ae
from utils.model import load_lenet
from utils.file import load_from_json


model_config = load_from_json('./model_config.json')
data_config = load_from_json('./data_config.json')
attack_config = load_from_json('./attack_config.json')

model_file = os.path.join(model_config.get("dir"), model_config.get("um_file"))

data_file = os.path.join(data_config.get('dir'), data_config.get('bs_file'))
data_bs = np.load(data_file)

label_file = os.path.join(data_config.get('dir'), data_config.get('label_file'))
labels = np.load(label_file)

target = load_lenet(file=model_file, wrap=True)

data_bs = data_bs[:10]
labels = labels[:10]
generate_ae(model=target, data=data_bs, labels=labels, attack_configs=attack_config)
exit()
