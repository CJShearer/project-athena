# parse configurations (into a dictionary) from json file
import os
import numpy as np

from tutorials.craft_adversarial_examples import generate_ae
from task1.attack2.my_attack import evaluate_models
from utils.data import subsampling
from utils.file import load_from_json
from utils.model import load_lenet
from matplotlib import pyplot as plt

model_configs = load_from_json("model-mnist.json")
data_configs = load_from_json("results/sample.json")
attack_configs = load_from_json("attack-zk-mnist.json")

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
# print(len(data_bs))

#def ratio(args):
   # pass
#print(len(data_bs))
#data_bs, labels = subsampling(data_bs, labels, 10,.001, filepath='results', filename='10')
# data_bs = data_bs[:10]
#labels = labels[:10]
#print(len(data_bs))
#generate_ae(model=target, data=data_bs, labels=labels, attack_configs=attack_configs, save=True, #output_dir="./results")
generate_ae(model=target, data=data_bs, labels=labels, attack_configs=attack_configs) #save=True, #output_dir="./results")

# parse configurations (into a dictionary) from json file
trans_configs = load_from_json("athena-mnist.json")
model_configs = load_from_json("model-mnist.json")
info_configs = load_from_json("./results/sample.json")
evaluate_models(trans_configs, model_configs, info_configs)
