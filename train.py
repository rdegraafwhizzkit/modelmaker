import os
import zipfile
import tensorflow as tf
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

assert tf.__version__.startswith('2')

image_path = tf.keras.utils.get_file(
    'flower_photos.tgz',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')
data = DataLoader.from_folder(image_path)

train_data, test_data = data.split(0.9)
model = image_classifier.create(train_data)
loss, accuracy = model.evaluate(test_data)
model.export(export_dir='./models/')

archive = zipfile.ZipFile('./models/model.tflite', 'r')
with open('./models/labels.txt', 'wb') as labels:
    labels.write(archive.read('labels.txt'))
