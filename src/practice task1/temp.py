from scripts.zhymir_scripts.my_attack import evaluate_models
from utils.file import load_from_json


trans_configs = load_from_json('../configs/task1/athena-mnist.json')
model_configs = load_from_json('md.json')
data_configs = load_from_json('result/dt2.json')
evaluate_models(trans_configs, model_configs, data_configs, save=False, output_dir='../../Task1/results')
