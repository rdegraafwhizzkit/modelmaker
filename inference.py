from PIL import Image
from os.path import expanduser
from pathlib import Path
from pycoral.adapters import common, classify
import tflite_runtime.interpreter as tflite

script_dir = Path(__file__).parent.absolute()
model_path = f'{script_dir}/models/model_edgetpu.tflite'
labels_path = f'{script_dir}/models/labels.txt'
image_file = f'{expanduser("~")}/.keras/datasets/flower_photos/daisy/102841525_bd6628ae3c.jpg'

labels = {i: label.strip() for i, label in enumerate(open(labels_path, 'r').readlines())}

interpreter = tflite.Interpreter(
    model_path,
    experimental_delegates=[tflite.load_delegate('libedgetpu.1.dylib')]
)
interpreter.allocate_tensors()
size = common.input_size(interpreter)

image = Image.open(image_file).convert('RGB').resize(size, Image.Resampling.LANCZOS)
common.set_input(interpreter, image)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)
for i, clazz in enumerate(classes):
    print(f'{i}: {labels.get(clazz.id)} / {clazz.score}')
