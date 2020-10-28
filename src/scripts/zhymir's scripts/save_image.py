import os
import time
import numpy as np


import matplotlib.pyplot as plt
from matplotlib import image

from utils.file import load_from_json
from utils.metrics import error_rate
from utils.model import load_lenet


def generate_images(model, data, labels, aes, attack_configs, min_img=1, show=True, save=False, output_dir=None):
    """
    Generate adversarial examples
    :param model: WeakDefense. The targeted model.
    :param data: array. The benign samples to generate adversarial for.
    :param labels: array or list. The true labels.
    :param attack_configs: dictionary. Attacks and corresponding settings.
    :param save: boolean. True, if save the images.
    :param output_dir: str or path. Location to save the adversarial examples.
        It cannot be None when save is True.
    :return:
    """
    img_rows, img_cols = data.shape[1], data.shape[2]
    num_attacks = attack_configs.get("num_attacks")

    if len(labels.shape) > 1:
        labels = np.asarray([np.argmax(p) for p in labels])

    # generate attacks one by one
    for id in range(num_attacks):
        key = "configs{}".format(id)
        data_adv = aes[id]
        # predict the adversarial examples
        predictions = model.predict(data_adv)
        predictions = np.asarray([np.argmax(p) for p in predictions])

        err = error_rate(y_pred=predictions, y_true=labels)
        print(">>> error rate:", err)

        # plotting some examples
        num_plotting = min(data.shape[0], min_img)
        for i in range(num_plotting):
            img = data_adv[i].reshape((img_rows, img_cols))
            plt.imshow(img, cmap='gray')
            title = '{}: {}->{}'.format(attack_configs.get(key).get("description"),
                                        labels[i],
                                        predictions[i]
                                        )
            plt.title(title)
            if save:
                if output_dir is None:
                    raise ValueError('output directory cannot be None if save is true')
                name = '{}_image_{}.png'.format(attack_configs.get(key).get("description"), i)
                filepath = os.path.join(output_dir, name)
                plt.savefig(filepath)
            if show:
                plt.show()
            plt.close()


# model_configs = load_from_json('./model-config.json')
# model_file = os.path.join(model_configs.get("dir"), model_configs.get("um_file"))
# target = load_lenet(file=model_file, wrap=True)
# data_file = load_from_json('./data-config.json')
# attack_file = load_from_json('./attack-config.json')
if __name__ == '__main__':
    # parse configurations (into a dictionary) from json file
    model_configs = load_from_json('../../../task1/attack2/model-config.json')
    data_configs = load_from_json('../../../task1/attack2/sub-data-config.json')
    attack_configs = load_from_json('../../../task1/attack2/attack-config.json')

    # load the targeted model
    model_file = os.path.join(model_configs.get("dir"), model_configs.get("um_file"))
    target = load_lenet(file=model_file, wrap=True)

    # load the benign samples
    data_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
    data_bs = np.load(data_file)
    # load the corresponding true labels
    label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
    labels = np.load(label_file)

    aes = [np.load(os.path.join(data_configs.get('dir'), x)) for x in data_configs.get('ae_files')]
    generate_images(target, data_bs, labels, aes, attack_configs, show=False, min_img=1, save=True, output_dir='../../../task1/attack2/images')
    # generate_images(target, data_file, attack_configs, min_img=3)