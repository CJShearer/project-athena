from utils.model import load_pool, load_lenet
from utils.file import load_from_json
from utils.metrics import error_rate, get_corrections
from models.athena import Ensemble, ENSEMBLE_STRATEGY

import os
import numpy as np


def collect_raw_prediction(trans_configs, model_configs, data_configs, use_logits=False, active_list=False):
    """

    :param trans_configs:
    :param model_configs:
    :param data_configs:
    :param use_logits: Boolean. If True, the model will return logits value (before ``softmax``),
                    return probabilities, otherwise.
    :param active_list: Boolean. If True, only the supplied list of active WDs will be used
    :return: 3D array of predictions
    """
    # load the pool and create the ensemble
    pool, wd_models = load_pool(trans_configs=trans_configs,
                        model_configs=model_configs,
                        active_list=active_list,
                        use_logits=use_logits,
                        wrap=True
                        )
    athena = Ensemble(classifiers=list(pool.values()),
                     strategy=ENSEMBLE_STRATEGY.MV.value)

    # load training/testing data
    print('>>> Loading benign and adversarial samples as training/testing data')
    bs_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
    ae_file_list = data_configs.get('ae_files')
    x = np.empty([len(ae_file_list)+1, 10000, 28, 28, 1])

    # load the benign samples
    x[0] = np.load(bs_file)

    # load the adversarial samples
    for i in range(len(ae_file_list)):
        ae_file = os.path.join(data_configs.get('dir'), ae_file_list[i])
        x[i+1] = np.load(ae_file)

    print('>>> Loaded samples as ndarray with shape', x.shape,
          ' => (sets of images, number of images in set, image width, image height, pixel value)')
    print('    (0, 10000, 28, 28, 1) is the set of benign samples')
    print('    (1:45, 10000, 28, 28, 1) are the sets of adversarial samples','\n')

    print('>>> Collecting raw predictions from', len(wd_models), 'models for', x.shape[0],
          'sets of', x.shape[1], 'images:')

    raw_preds = np.empty([x.shape[0], len(wd_models), x.shape[1], 10])
    for i in range(x.shape[0]):
        print('   collecting raw predictions for set ', i)
        raw_preds[i] = (athena.predict(x=x[i], raw=True))
    return raw_preds


if __name__ == '__main__':
    # load experiment configurations
    trans_configs = load_from_json("../../configs/task2/cody_configs/athena-mnist.json")
    model_configs = load_from_json("../../configs/task2/cody_configs/model-mnist.json")
    data_configs = load_from_json("../../configs/task2/cody_configs/data-mnist.json")
    output_dir = "../../../Task2/data"

    # collect the predictions
    raw_preds = collect_raw_prediction(trans_configs=trans_configs,
                                       model_configs=model_configs,
                                       data_configs=data_configs,
                                       use_logits=True,
                                       active_list=True)

    file = os.path.join(output_dir, "predictions.npz")
    print('>>> Saving compressed predictions to ', file)
    np.savez_compressed(file, raw_preds)
    print('>>> Predictions saved. You may now close the terminal.')