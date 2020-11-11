import os

from utils.file import load_from_json
from utils.model import load_lenet
import numpy as np
from utils.model import load_pool
from utils.metrics import error_rate, get_corrections
from models.athena import Ensemble, ENSEMBLE_STRATEGY


def generate_ae(model, data, labels, attack_configs, save=False, output_dir=None):
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

        # plotting some examplesjsm
        num_plotting = min(data.shape[0], 2)
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
            # save with a random name
            file = os.path.join(output_dir, "{}.npy".format(time.monotonic()))
            print("Save the adversarial examples to file [{}].".format(file))
            np.save(file, data_adv)

model_configs = load_from_json("./md.json")
data_configs = load_from_json("./dt.json")
attack_configs = load_from_json("./at.json")

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

# copied from tutorials/eval_model.py
def evaluate(trans_configs, model_configs,
             data_configs, save=False, output_dir=None):
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
    # Load the baseline defense (PGD-ADT model)
    baseline = load_lenet(file=model_configs.get('jsma_trained'), trans_configs=None,
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
    ae_file = os.path.join(data_configs.get('dir'), ae_list[4])
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
    results['JSMA-ADT'] = err_bl

    # TODO: collect and dump the evaluation results to file(s) such that you can analyze them later.
    print(">>> Evaluations on [{}]:\n{}".format(ae_file, results))

    attack_configs = load_from_json("../src/configs/demo/at.json")
    model_configs = load_from_json("../src/configs/demo/md.json")
    data_configs = load_from_json("../src/configs/demo/dt.json")

    output_dir = "../results"

    # evaluate
    evaluate(attack_configs=trans_configs,
             model_configs=model_configs,
             data_configs=data_configs,
             save=False,
             output_dir=output_root)
