import matplotlib.pyplot as plt
import numpy as np
import keras
from art.classifiers.classifier import Classifier, ClassifierNeuralNetwork, ClassifierGradients
from keras.callbacks import ModelCheckpoint
from attacks.attack import generate
from models.athena import Ensemble
from models.keraswrapper import WeakDefense
from utils.data import get_dataloader
from utils.file import dump_to_json, load_from_json
from utils.model import load_pool

def plot_model_figs(history_p, show=True, save=False, save_name=None):
    """ Plots model metrics with added ability to save.
        history_p is history.history object returned from model.fit or model.evaluate
        save_name: the name for the base, function adds postfix and .png end"""
    if 'categorical_accuracy':
        plt.plot(history_p['categorical_accuracy'])
        plt.plot(history_p['val_categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if save and save_name:
            plt.savefig(save_name+'_acc.png')
        if show:
            plt.show()
    if 'accuracy' in history_p:
        plt.plot(history_p['accuracy'])
        plt.plot(history_p['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if save and save_name:
            plt.savefig(save_name+'_acc.png')
        if show:
            plt.show()

    if 'loss' in history_p:
        plt.plot(history_p['loss'])
        plt.plot(history_p['val_loss'])
        # plt.xlim((0, 1))
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if save and save_name:
            plt.savefig(save_name+'_loss.png')
        if show:
            plt.show()


def add_checkpoint(filepath_p, callback_list):
    """ Adds checkpoints to model"""
    checkpoint = ModelCheckpoint(filepath_p+'_checkpoint', monitor='loss', verbose=1, save_best_only=False, mode='min')
    # callback_list.extend([checkpoint])
    callback_list = [checkpoint]


def make_ensemble(wd_config, model_config):
    """ Creates ensemble from weak defense config and model config"""
    pool, _ = load_pool(trans_configs=wd_config, model_configs=model_config, active_list=True)
    # turns WD into WeakDefense objects for Ensemble
    wd_models = [WeakDefense(pool[wd], wd_config.get('configs'+str(wd))) for wd in pool]
    wds = Ensemble(wd_models, strategy=None)
    return wds


def combine_history(history_1, history_2):
    """ Takes two history object history dictionaries
     and returns combined dictionary"""
    if history_1.keys() != history_2.keys() and history_1 and history_1.keys():
        print("keys are not the same")
        return
    # assuming they're lists
    combined = {key: (history_1[key]+history_2[key]) if key in history_1 else history_2[key] for key in history_1}
    return combined

# class inner_model(ClassifierNeuralNetwork, Classifier):
#     def __init__(self):
#         def predict(self, x, **kwargs):  # lgtm [py/inheritance/incorrect-overridden-signature]
#             """
#             Perform prediction of the classifier for input `x`.
#
#             :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
#                       nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2)
#             :type x: `np.ndarray`
#             :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
#             :rtype: `np.ndarray`
#             """
#             raise NotImplementedError
#
#         @abc.abstractmethod
#         def fit(self, x, y, **kwargs):  # lgtm [py/inheritance/incorrect-overridden-signature]
#             """
#             Fit the classifier using the training data `(x, y)`.
#
#             :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
#                       nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2)
#             :type x: `np.ndarray`
#             :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
#                       (nb_samples,).
#             :type y: `np.ndarray`
#             :param kwargs: Dictionary of framework-specific arguments.
#             :type kwargs: `dict`
#             :return: `None`
#             """
#             raise NotImplementedError
#         @abc.abstractmethod
#         def nb_classes(self):
#             """
#             Return the number of output classes.
#
#             :return: Number of classes in the data.
#             :rtype: `int`
#             """
#             raise NotImplementedError
#
#         @abc.abstractmethod
#         def save(self, filename, path=None):
#             """
#             Save a model to file specific to the backend framework.
#
#             :param filename: Name of the file where to save the model.
#             :type filename: `str`
#             :param path: Path of the directory where to save the model. If no path is specified, the model will be stored in
#                          the default data location of ART at `ART_DATA_PATH`.
#             :type path: `str`
#             :return: None
#             """
#             raise NotImplementedError
#
#         @abc.abstractmethod
#         def get_activations(self, x, layer, batch_size):
#             """
#             Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
#             `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
#             calling `layer_names`.
#
#             :param x: Input for computing the activations.
#             :type x: `np.ndarray`
#             :param layer: Layer for computing the activations
#             :type layer: `int` or `str`
#             :param batch_size: Size of batches.
#             :type batch_size: `int`
#             :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
#             :rtype: `np.ndarray`
#             """
#             raise NotImplementedError
#
#         @abc.abstractmethod
#         def set_learning_phase(self, train):
#             """
#             Set the learning phase for the backend framework.
#
#             :param train: `True` if the learning phase is training, `False` if learning phase is not training.
#             :type train: `bool`
#             """
#             raise NotImplementedError

class athena_with_model():
    """ Class that combines Ensemble with model for training"""
    def __init__(self, wd_config, model_config, model):
        """
        Takes json file or path for Weak defense config and model config
        Takes model as keras model.
        """
        if isinstance(wd_config, str):
            wd_config = load_from_json(wd_config)
        if isinstance(model_config, str):
            model_config = load_from_json(model_config)
        self._ensemble = make_ensemble(wd_config, model_config)
        # self._model = WeakDefense(model, None)
        # self._model = ClassifierNeuralNetwork(model)
        self._model = model
        self._callbacks = []
        self._history = dict()

    def fit(self, input_images, labels, batch_size=128, load_progress=False, save_history=True, save_progress=False, checkpoint_path=None):
        print("fitting")
        wd_predictions = self.__get_wd_predictions__(input_images)
        history = None
        if load_progress:
            history = self._model.fit(wd_predictions, labels, batch_size=batch_size)
        else:
            history = self._model.fit(wd_predictions, labels, batch_size=batch_size)
        if save_progress and checkpoint_path is not None:
            checkpoint = ModelCheckpoint(checkpoint_path + '_checkpoint', monitor='loss', verbose=1, save_best_only=False, mode='min')
            self._callbacks.append(checkpoint)
        if save_history:
            self._history = history.history if history is not None else self._history
        return history.history

    def predict(self, input_images):
        self._model.predict(self.__get_wd_predictions__(input_images))

    def evaluate(self, test, labels):
        print('evaluating')
        # keras.models.Sequential.evaluate()
        return self._model.evaluate(self.__get_wd_predictions__(test), labels, batch_size=len(test))

    def save_model(self, filename):
        self._model.save(filename)

    def train_adversarial_pgd(self, input_images, labels, epochs=1, eps=0.3, eps_step=1/10, max_iter=10, norm='l2', targeted=False, num_random_init=0, random_eps=False):
        attack_args = {'attack': 'pgd', 'description': 'none', 'eps': eps, 'eps_step': eps_step, 'max_iter': max_iter,
                       'norm': norm, 'targeted': targeted, 'num_random_init': num_random_init, 'random_eps': random_eps}
        data_loader = get_dataloader(input_images, labels, batch_size=1000, shuffle=True)
        for epoch in range(epochs):
            print(epoch)
            itr = iter(data_loader)
            for item in itr:
                print('generating data')
                data_adv = generate(self._ensemble, item, attack_args=attack_args)
                print('data generated, fitting to adversarial')
                history = self.fit(data_adv, item[1], batch_size=10, load_progress=True, save_history=False, save_progress=True, checkpoint_path='new_ensemble_checkpoint')
                self._history = combine_history(self._history, history)

    def save_history(self, filename):
        dump_to_json(self._history, filename+'_history')

    def __get_wd_predictions__(self, input_images, transposed=True):
        """ Calculates wd predictions and adds param for transposition"""
        wd_predictions = self._ensemble.predict(input_images, batch_size=len(input_images), raw=True)
        return np.transpose(wd_predictions, (1, 0, 2)) if transposed else wd_predictions

# class athena_with_model(Ensemble):
#     self.__init__():
#     eps = attack_args.get('eps', 0.3)
#     eps_step = attack_args.get('eps_step', eps/10.)
#     max_iter = attack_args.get('max_iter', 10)
#
#     norm = _get_norm_value(attack_args.get('norm', 'linf'))
#     targeted = attack_args.get('targeted', False)
#     num_random_init = attack_args.get('num_random_init', 0)
#     random_eps = attack_args.get('random_eps', False)

# class ensemble_nn(Ensemble):
#     def __init__(self, classifiers, strategy, model, classifier_weights=None,
#                  channel_index=3, clip_values=(0., 1.), preprocessing_defences=None,
#                  postprocessing_defences=None, preprocessing=(0, 1)):
#         super.__init__(self, classifiers, strategy, classifier_weights=classifier_weights,
#                  channel_index=channel_index, clip_values=clip_values, preprocessing_defences=preprocessing_defences,
#                  postprocessing_defences=postprocessing_defences, preprocessing=preprocessing)
#         self._model = model
