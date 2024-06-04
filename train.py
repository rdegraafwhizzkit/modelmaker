import zipfile
import tensorflow as tf
# from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
# from tflite_model_maker import image_classifier
# from tflite_model_maker.image_classifier import DataLoader
from pathlib import Path
from logging import basicConfig, INFO

basicConfig(level=INFO)

assert tf.__version__.startswith('2')

script_dir = Path(__file__).parent.absolute()
export_dir='./models/'
image_path = f'{script_dir}/data/matchit/zoo'

# data = DataLoader.from_folder(image_path)
# train_data, test_data = data.split(0.9)
# model = image_classifier.create(train_data)
# # loss, accuracy = model.evaluate(test_data)
# model.export(
#     export_dir=export_dir,
#     export_format=ExportFormat.SAVED_MODEL
# )

# Convert the saved model using TFLiteConverter
converter = tf.lite.TFLiteConverter.from_saved_model(f'{export_dir}/saved_model')
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
with open(f'{export_dir}model.tflite', 'wb') as tflite_model:
    tflite_model.write(converter.convert())

archive = zipfile.ZipFile('./models/model.tflite', 'r')
with open('./models/labels.txt', 'wb') as labels:
    labels.write(archive.read('labels.txt'))

