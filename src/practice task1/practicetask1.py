import os

from tutorials.craft_adversarial_examples import generate_ae
from utils.file import load_from_json
from utils.model import load_lenet
import numpy as np


model_configs = load_from_json("md.json")
<<<<<<< HEAD
data_configs = load_from_json("dt2.json")
=======
data_configs = load_from_json("result/dt2.json")
<<<<<<< HEAD
>>>>>>> 6ec7374720c91fd8e3d1b402b37bd364b7a7a8db
=======
>>>>>>> 5007ebbb5f5c855d164aecf8c0d82ae725532b9b
>>>>>>> eb83fac7eb5cfc2cdcd38ab0de37595583db9943
attack_configs = load_from_json("at.json")

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
# data_bs = data_bs[:10]
# labels = labels[:10]
generate_ae(model=target, data=data_bs, labels=labels, attack_configs=attack_configs, save=True, output_dir="./result")
