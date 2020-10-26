import os

from attacks.attack import generate
# from tutorials.craft_adversarial_examples import generate_ae
from tutorials.eval_model import evaluate
from utils.data import subsampling
# parse configurations (into a dictionary) from json file
from utils.file import load_from_json
from utils.metrics import error_rate
from utils.model import load_lenet
import numpy as np
import matplotlib as plt
import time


# copied from tutorial modified to use names in data config
def generate_ae_with_names(model, data, labels, attack_configs, save=False, output_dir=None, filenames=None):
    """
    Generate adversarial examples
    :param model: WeakDefense. The targeted model.
    :param data: array. The benign samples to generate adversarial for.
    :param labels: array or list. The true labels.
    :param attack_configs: dictionary. Attacks and corresponding settings.
    :param save: boolean. True, if save the adversarial examples.
    :param output_dir: str or path. Location to save the adversarial examples.
        It cannot be None when save is True.
    :return:
    """
    img_rows, img_cols = data.shape[1], data.shape[2]
    num_attacks = attack_configs.get("num_attacks")
    data_loader = (data, labels)

    if len(labels.shape) > 1:
        labels = np.asarray([np.argmax(p) for p in labels])

    # generate attacks one by one
    for id in range(num_attacks):
        key = "configs{}".format(id)
        data_adv = generate(model=model,
                            data_loader=data_loader,
                            attack_args=attack_configs.get(key)
                            )
        # predict the adversarial examples
        predictions = model.predict(data_adv)
        predictions = np.asarray([np.argmax(p) for p in predictions])

        err = error_rate(y_pred=predictions, y_true=labels)
        print(">>> error rate:", err)

        # plotting some examples
        num_plotting = min(data.shape[0], 0)
        for i in range(num_plotting):
            img = data_adv[i].reshape((img_rows, img_cols))
            plt.imshow(img, cmap='gray')
            title = '{}: {}->{}'.format(attack_configs.get(key).get("description"),
                                        labels[i],
                                        predictions[i]
                                        )
            plt.title(title)
            plt.show()
            plt.close()

        # save the adversarial example
        if save:
            if output_dir is None:
                raise ValueError("Cannot save images to a none path.")
            if filenames is None or len(filenames) < num_attacks:
                # save with a random name
                file = os.path.join(output_dir, "{}.npy".format(time.monotonic()))
                print("Save the adversarial examples to file [{}].".format(file))
                np.save(file, data_adv)
            else:
                # save with a file name
                file = os.path.join(output_dir, filenames[id])
                print("Save the adversarial examples to file [{}].".format(file))
                np.save(file, data_adv)


def my_attack(model_config, data_config, attack_config, ratio=0.1):
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
    print(len(data_bs))
    data_bs, labels = subsampling(data_bs, labels, 10, ratio, filepath='results', filename='10')
    # data_bs = data_bs[:100]
    # labels = labels[:100]
    print(len(data_bs))
    generate_ae_with_names(model=target, data=data_bs, labels=labels, attack_configs=attack_configs, save=True, output_dir="./results", filenames=data_configs.get('ae_files'))


# my_attack('model-config.json', 'data-config.json', 'attack-config.json', ratio=0.001)
trans_configs = load_from_json('athena-mnist.json')
model_configs = load_from_json('model-config.json')
data_configs = load_from_json('./results/sub-data-config.json')
evaluate(trans_configs, model_configs, data_configs)
