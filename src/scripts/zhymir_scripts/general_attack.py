import os
import numpy as np
# from Task1.attack2 import my_attack

# call my attack with my_attack(model_config name, data_config name, attack_config name)
from tutorials.craft_adversarial_examples import generate_ae
from utils.data import subsampling
from utils.file import load_from_json
from utils.model import load_lenet


def gen_attack(model_config, data_config, attack_config, ratio=0.1, output_dir=None):
    model_configs = load_from_json(model_config)
    data_configs = load_from_json(data_config)
    attack_configs = load_from_json(attack_config)

    # load the targeted model
    model_file = os.path.join(model_configs.get("dir"), model_configs.get("um_file"))
    target = load_lenet(file=model_file, wrap=True)

    # load the benign samples
    data_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
    data_bs = np.load(data_file)
    # load the corresponding true labels
    label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
    labels = np.load(label_file)

    # generate adversarial examples for a small subset
    data_bs, labels = subsampling(data_bs, labels, 10, ratio)
    # data_bs = data_bs[:100]
    # labels = labels[:100]
    generate_ae(model=target, data=data_bs, labels=labels, attack_configs=attack_configs)#, save=True, output_dir=output_dir)


gen_attack('./attack2/model-config.json', './attack2/results/sub-data-config.json', './attack2/attack-config.json')
# gen_attack('model-config.json', 'data-config.json', 'attack-config.json', ratio=0.2, output_dir="./results")
