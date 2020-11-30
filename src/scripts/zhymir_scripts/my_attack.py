import os
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

from attacks.attack import generate
# from tutorials.craft_adversarial_examples import generate_ae
from models.athena import ENSEMBLE_STRATEGY, Ensemble
from utils.data import subsampling
# parse configurations (into a dictionary) from json file
from utils.file import load_from_json
from utils.metrics import error_rate, get_corrections
from utils.model import load_lenet, load_pool


# copied from tutorial modified to use names in data config
def generate_ae_with_names(model, data, labels, attack_configs, save=False, output_dir=None, filenames=None, save_img=True, show=True, num_img=3, img_output=None):
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
        num_plotting = min(data.shape[0], num_img)
        for i in range(num_plotting):
            img = data_adv[i].reshape((img_rows, img_cols))
            plt.imshow(img, cmap='gray')
            title = '{}: {}->{}'.format(attack_configs.get(key).get("description"),
                                        labels[i],
                                        predictions[i]
                                        )
            plt.title(title)
            if save_img:
                if img_output is None:
                    raise ValueError('output directory cannot be None if save is true')
                name = '{}_image_{}.png'.format(attack_configs.get(key).get("description"), i)
                filepath = os.path.join(output_dir, name)
                plt.savefig(filepath)
            if show:
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


def my_attack(model_config, data_config, attack_config, generate_sub=False, ratio=0.1, sub_data_path=None,
              sub_data_name=None, result_path=None, save_img=True, show=True, img_output=None):
    model_configs = load_from_json(model_config) if not isinstance(model_config, dict) else model_config
    data_configs = load_from_json(data_config) if not isinstance(data_config, dict) else data_config
    attack_configs = load_from_json(attack_config) if not isinstance(attack_config, dict) else attack_config

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
    if generate_sub:
        if sub_data_name and sub_data_path:
            data_bs, labels = subsampling(data_bs, labels, 10, ratio, filepath=sub_data_path, filename=sub_data_name)
        else:
            data_bs, labels = subsampling(data_bs, labels, 10, ratio)
    # data_bs = data_bs[:100]
    # labels = labels[:100]
    # print(len(data_bs))
    generate_ae_with_names(model=target, data=data_bs, labels=labels, attack_configs=attack_configs, save=True,
                           output_dir=result_path, filenames=data_configs.get('ae_files'),
                           save_img=save_img, show=show, img_output=img_output)


def evaluate_models(trans_configs, model_configs,
             data_configs, save=False, output_dir=None, filenames=None):
    """
    Apply transformation(s) on images.
    :param trans_configs: dictionary. The collection of the parameterized transformations to test.
        in the form of
        { configsx: {
            param: value,
            }
        }
        The key of a configuration is 'configs'x, where 'x' is the id of corresponding weak defense.
    :param model_configs:  dictionary. Defines model related information.
        Such as, location, the undefended model, the file format, etc.
    :param data_configs: dictionary. Defines data related information.
        Such as, location, the file for the true labels, the file for the benign samples,
        the files for the adversarial examples, etc.
    :param save: boolean. Save the transformed sample or not.
    :param output_dir: path or str. The location to store the transformed samples.
        It cannot be None when save is True.
    :return:
    """
    model_configs = load_from_json(model_configs) if not isinstance(model_configs, dict) else model_configs
    data_configs = load_from_json(data_configs) if not isinstance(data_configs, dict) else data_configs
    trans_configs = load_from_json(trans_configs) if not isinstance(trans_configs, dict) else trans_configs
    # Load the baseline defense (PGD-ADT model)
    baseline = load_lenet(file=model_configs.get('pgd_trained'), trans_configs=None,
                                  use_logits=False, wrap=False)

    # get the undefended model (UM)
    file = os.path.join(model_configs.get('dir'), model_configs.get('um_file'))
    undefended = load_lenet(file=file,
                            trans_configs=trans_configs.get('configs0'),
                            wrap=True)
    print(">>> um:", type(undefended))

    # load weak defenses into a pool
    pool, _ = load_pool(trans_configs=trans_configs,
                        model_configs=model_configs,
                        active_list=True,
                        wrap=True)
    # create an AVEP ensemble from the WD pool
    wds = list(pool.values())
    print(">>> wds:", type(wds), type(wds[0]))
    ensemble = Ensemble(classifiers=wds, strategy=ENSEMBLE_STRATEGY.AVEP.value)

    # load the benign samples
    bs_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
    x_bs = np.load(bs_file)
    img_rows, img_cols = x_bs.shape[1], x_bs.shape[2]

    # load the corresponding true labels
    label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
    labels = np.load(label_file)

    # get indices of benign samples that are correctly classified by the targeted model
    print(">>> Evaluating UM on [{}], it may take a while...".format(bs_file))
    pred_bs = undefended.predict(x_bs)
    corrections = get_corrections(y_pred=pred_bs, y_true=labels)

    # Evaluate AEs.
    results = {}
    ae_list = data_configs.get('ae_files')
    for index in range(len(ae_list)):
        ae_file = os.path.join(data_configs.get('dir'), ae_list[index])
        x_adv = np.load(ae_file)

        # evaluate the undefended model on the AE
        print(">>> Evaluating UM on [{}], it may take a while...".format(ae_file))
        pred_adv_um = undefended.predict(x_adv)
        err_um = error_rate(y_pred=pred_adv_um, y_true=labels, correct_on_bs=corrections)
        # track the result
        results['UM'] = err_um

        # evaluate the ensemble on the AE
        print(">>> Evaluating ensemble on [{}], it may take a while...".format(ae_file))
        pred_adv_ens = ensemble.predict(x_adv)
        err_ens = error_rate(y_pred=pred_adv_ens, y_true=labels, correct_on_bs=corrections)
        # track the result
        results['Ensemble'] = err_ens

        # evaluate the baseline on the AE
        print(">>> Evaluating baseline model on [{}], it may take a while...".format(ae_file))
        pred_adv_bl = baseline.predict(x_adv)
        err_bl = error_rate(y_pred=pred_adv_bl, y_true=labels, correct_on_bs=corrections)
        # track the result
        results['PGD-ADT'] = err_bl

        # TODO: collect and dump the evaluation results to file(s) such that you can analyze them later.
        print(">>> Evaluations on [{}]:\n{}".format(ae_file, results))
        test_result = ">>> Evaluations on [{}]:\n{}".format(ae_file, results)
        if save:
            if output_dir is None:
                raise ValueError("Cannot save images to a none path.")
            if filenames is None or len(filenames) < len(ae_list):
                # save with a random name
                name = ae_list[index].split('.npy')[0]+'_eval.csv'
                file = os.path.join(output_dir, "{}.".format(name))
                print("Save the adversarial examples to file [{}].".format(file))
                with open(file, 'w') as csv_file:
                    fieldnames = list(results.keys())
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(results)
                # np.save(file, test_result)
            else:
                # save with a file name
                file = os.path.join(output_dir, filenames[index])
                print("Save the adversarial examples to file [{}].".format(file))
                writer = csv.writer(open(file, 'w'))
                writer.writerow(test_result)
                # np.save(file, test_result)


if __name__ == '__main__':
    config_root = '../../configs/task3'
    result_root = '../../../Task3/results'
    image_root = '../../../Task3/images'
    model_configs = os.path.join(config_root, 'model_config.json')
    data_configs = os.path.join(config_root, 'data_cw_config.json')
    attack_configs = os.path.join(config_root, 'attack_2_cw_config.json')
    trans_configs = os.path.join(config_root, 'athena-mnist.json')
    sub_data_path = '../../../Task3/data'
    sub_data_name = '1000'
    # 'subsamples-{}-ratio_{}-{}.npy'
    # generate adversarial examples for a small subset
    # generate_ae_with_names(target, data_bs, labels, attack_configs)
    # my_attack(model_configs, data_configs, attack_configs, ratio=0.1, sub_data_path=sub_data_path,
    #           sub_data_name=sub_data_name, result_path=result_root, save_img=True, show=False, img_output=image_root)
    # exit()
    # sub_data_config = os.path.join(config_root, 'sub_data_config.json')
    evaluate_models(trans_configs, model_configs, data_configs, save=True, output_dir='../../../Task3/results')
