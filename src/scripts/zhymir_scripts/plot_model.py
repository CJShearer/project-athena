import os

import matplotlib.pyplot as plt
from keras.models import load_model
from utils.file import load_from_json


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


if __name__ == '__main__':
    data_root = '../../../Task2/data'
    model_root = '../../../Task2/models'
    # model_path = os.path.join(model_root, 'zhymir_model_4_layer.h5')
    # model = load_model(model_path, compile=False)
    history = load_from_json(os.path.join(data_root, 'zhymir_model_batch_8_4_layer_history'))
    plot_model_figs(history)
    print(history)
    # exit()
    # if 'accuracy' in history:
    #     plt.plot(history['accuracy'])
    #     plt.plot(history['val_accuracy'])
    #     plt.title('model accuracy')
    #     plt.ylabel('accuracy')
    #     plt.xlabel('epoch')
    #     plt.legend(['train', 'test'], loc='upper left')
    #     plt.show()
    #
    # if 'loss' in history:
    #     plt.plot(history['loss'])
    #     plt.plot(history['val_loss'])
    #     # plt.xlim((0, 1))
    #     plt.title('model loss')
    #     plt.ylabel('loss')
    #     plt.xlabel('epoch')
    #     plt.legend(['train', 'test'], loc='upper left')
    #     plt.show()