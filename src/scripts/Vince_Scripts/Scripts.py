from scripts.zhymir_scripts.my_attack import *
if __name__ == '__main__':
    config_root = '../../configs/task3/'
    result_root = '../../../Task3/results'
    model_configs = os.path.join(config_root, 'model_config.json')
    data_configs = os.path.join(config_root, 'pgd_data.json')
    attack_configs = os.path.join(config_root, 'attack-zk-mnist.json')
    trans_configs = os.path.join(config_root, 'athena-mnist.json')
    sub_data_path = '../../../Task3/data'
    sub_data_name = '1000'
    # 'subsamples-{}-ratio_{}-{}.npy'
    # generate adversarial examples for a small subset
    # generate_ae_with_names(target, data_bs, labels, attack_configs)
    # my_attack(model_configs, data_configs, attack_configs, result_path=result_root)
    # exit()
    sub_data_config = os.path.join(config_root, 'sub_data_config.json')
    evaluate_models(trans_configs, model_configs, sub_data_config, save=False, output_dir='../../../Task3/results')