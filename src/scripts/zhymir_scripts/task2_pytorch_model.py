""" PyTorch implementation of Keras model
    Main purpose for gpu accelerated training"""
from abc import ABC

import torch
import torch.nn as nn

from collections import OrderedDict


class torchModel(ABC, nn.Module):
    """ pytorch complement to keras model"""
    def __init__(self):
        self._layers = OrderedDict([
            nn.Linear()
        ])
        self._model = nn.Sequential()

# model = keras.models.Sequential([
#     keras.layers.InputLayer(input_shape=(wds._nb_classifiers, 10), name='WD_layer'),
#     keras.layers.Flatten(),
#     keras.layers.Dense(units=100, activation='relu', name='D1'),
#     keras.layers.Dense(10, name='output_layer', activation='softmax')
# ])
# model.compile('adam', 'categorical_crossentropy')