import os

from keras.metrics import CategoricalAccuracy
from keras.models import load_model
import numpy as np
import pandas as pd

from utils.file import load_from_json


def evaluate_ensemble_model(model_path_and_filename, results_path_and_filename=None, save=False):

  # load and compile trained model
  model = load_model(model_path_and_filename, compile=False)
  metrics = [CategoricalAccuracy(dtype='float64')]
  model.compile('adam', 'categorical_crossentropy', metrics=metrics)

  # load configs
  model_config = load_from_json('../../configs/task2/cody_configs/model-mnist.json')
  data_config = load_from_json('../../configs/task2/cody_configs/data-mnist.json')
  WD_config = load_from_json('../../configs/task2/cody_configs/athena-mnist.json')

  # get list of AEs used
  ae_files = data_config.get('ae_files')

  # get labels, which are used for each set of 10k predictions
  labels = np.load(os.path.join(data_config.get('dir'), data_config.get('label_file')))
  # get pre-computed predictions of WDs and extract only AEs for testing
  test_data = np.transpose(np.load('../../../Task2/data/predictions.npz')['arr_0'], (0, 2, 1, 3))[1:len(ae_files)+1]
  # evaluate on each AE
  results = pd.DataFrame(index=model.metrics_names, columns=ae_files)
  for i in range(test_data.shape[0]):
    ae_file = ae_files[i]
    print('>>> Evaluating', ae_file)
    result = model.evaluate(test_data[i], labels)
    print('>>> Result', result)
    results[ae_file] = result

  # transpose results for more readable table
  results = results.T
  print(results.head(5))

  # save results
  if save:
    print('>>> Saving results to', results_path_and_filename)
    results.to_csv(results_path_and_filename)


if __name__ == '__main__':
  # path and file to save results
  results_path_and_filename = '../../../Task2/results/cody_model.csv'

  # path and file for model to be evaluated
  model_path_and_filename = '../../../Task2/models/cody_model.h5'

  evaluate_ensemble_model(model_path_and_filename, results_path_and_filename, save=False)
