from scripts.zhymir_scripts.task2_functions import evaluate_AEs, athena_with_model
import keras
from keras.models import load_model

from utils.file import load_from_json
WD_config = load_from_json('../../configs/task2/cody_configs/athena-mnist.json')
model_config = load_from_json('../../configs/task2/zhymir_configs/model-config.json')
model = keras.models.Sequential([
    keras.layers.Dense(units=100, input_shape=(16, 10), activation='relu', name='D1'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, name='output_layer', activation='softmax')
 ])
model.compile('adam', 'categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy(dtype='float64')])
#model = load_model(model_name, compile=False)
model.compile('adam', 'categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy(dtype='float64')])
new_ensemble = athena_with_model(WD_config, model_config, model)

evaluate_AEs(new_ensemble, load_from_json("../../configs/task2_update/new_config.json"))

