from PIL import Image
from os.path import expanduser
from pycoral.adapters import common, classify
import tflite_runtime.interpreter as tflite
from sys import platform
from ctypes.util import find_library
from dynaconf import Dynaconf
from logging import info, warning, basicConfig, INFO

basicConfig(level=INFO)
settings = Dynaconf(settings_files=['settings.yaml'], environments=True)

library = settings.libraries.get(platform)

experimental_delegates = []
model_path = settings.model.regular
if find_library(library) and settings.use_edgetpu:
    try:
        info('Trying to load EdgeTPU libraries')
        experimental_delegates = [tflite.load_delegate(library)]
        model_path = settings.model.edgetpu
        info('EdgeTPU libraries loaded')
    except ValueError:
        warning('EdgeTPU hardware not found, using CPU')
else:
    info('Configured not to use EdgeTPU' if not settings.use_edgetpu else 'EdgeTPU library not found')

labels_path = settings.model.labels
labels = {i: label.strip() for i, label in enumerate(open(labels_path, 'r').readlines())}
info(f'Loaded {len(labels)} labels')

info(f'Creating interpreter')
interpreter = tflite.Interpreter(
    model_path,
    experimental_delegates=experimental_delegates
)
interpreter.allocate_tensors()

info(f'Preparing input image')
size = common.input_size(interpreter)
image_file = f'{expanduser("~")}/.keras/datasets/flower_photos/daisy/102841525_bd6628ae3c.jpg'
image = Image.open(image_file).convert('RGB').resize(size, Image.Resampling.LANCZOS)

info(f'Inferencing')
common.set_input(interpreter, image)
interpreter.invoke()

info(f'Detected objects are:')
classes = classify.get_classes(interpreter)#, top_k=1)
for i, clazz in enumerate(classes):
    info(f'id: {i}, label: {labels.get(clazz.id)}, score: {clazz.score}')
