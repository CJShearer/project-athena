import os
import numpy as np

from utils.data import subsampling
from utils.file import load_from_json


data_configs = load_from_json('dt.json')
# load the benign samples
data_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
data_bs = np.load(data_file)
# load the corresponding true labels
label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
labels = np.load(label_file)

subsampling(data_bs, labels, 10, 0.001, './result', 'sub')