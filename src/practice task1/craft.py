import argparse
import numpy as np
import os
import time
from tutorials.craft_adversarial_examples import generate_ae
from matplotlib import pyplot as plt

from utils.model import load_lenet
from utils.file import load_from_json
from utils.metrics import error_rate
from attacks.attack import generate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-m', '--model-configs', required=False,
                        default='md.json',
                        help='Folder where models stored in.')
    parser.add_argument('-d', '--data-configs', required=False,
                        default='dt.json',
                        help='Folder where test data stored in.')
    parser.add_argument('-a', '--attack-configs', required=False,
                        default='at.json',
                        help='Folder where test data stored in.')
    parser.add_argument('-o', '--output-root', required=False,
                        default='results',
                        help='Folder for outputs.')
    parser.add_argument('--debug', required=False, default=True)

    args = parser.parse_args()

    print("------AUGMENT SUMMARY-------")
    print("MODEL CONFIGS:", args.model_configs)
    print("DATA CONFIGS:", args.data_configs)
    print("ATTACK CONFIGS:", args.attack_configs)
    print("OUTPUT ROOT:", args.output_root)
    print("DEBUGGING MODE:", args.debug)
    print('----------------------------\n')

    # parse configurations (into a dictionary) from json file
    model_configs = load_from_json(args.model_configs)
    data_configs = load_from_json(args.data_configs)
    attack_configs = load_from_json(args.attack_configs)

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
    data_bs = data_bs[:10]
    labels = labels[:10]
    generate_ae(model=target, data=data_bs, labels=labels, attack_configs=attack_configs)
