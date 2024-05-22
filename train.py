import os
import zipfile
import numpy as np
from pathlib import Path
import tensorflow as tf
from tflite_model_maker import model_spec, image_classifier
from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt
import matplotlib

assert tf.__version__.startswith('2')

matplotlib.use('Qt5Agg')

image_path = tf.keras.utils.get_file(
    'flower_photos.tgz',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')

CLASS_NAMES = np.array(
    [item.name for item in Path(image_path).glob('*') if item.name != 'LICENSE.txt']
)
print('These are the available classes:', CLASS_NAMES)

# raise Exception
data = DataLoader.from_folder(image_path)

train_data, test_data = data.split(0.9)

model = image_classifier.create(train_data)
# loss, accuracy = model.evaluate(test_data)
model.export(export_dir='./models/')

# Extract labels

archive = zipfile.ZipFile('./models/model.tflite', 'r')
with open('./models/labels.txt', 'wb') as labels:
    labels.write(archive.read('labels.txt'))

